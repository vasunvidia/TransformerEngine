# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Internal function used by multiple modules."""

import collections
import dataclasses
import queue
from typing import Any, Callable, List, Optional, Tuple, Union

import torch

from .. import cpp_extensions as tex
from ..constants import TE_DType
from ..distributed import in_fp8_activation_recompute_phase
from ..export import is_in_onnx_export_mode
from ..quantization import FP8GlobalStateManager
from ..tensor.hybrid_tensor import HybridQuantizer
from ..utils import get_default_init_method


def sum_bias_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce all token dimensions, preserving the final feature dimension."""
    if tensor.ndim == 1:
        return tensor.clone()
    return tensor.sum(dim=tuple(range(tensor.ndim - 1)))


def set_quantizer_amax_reduction_group(quantizer, amax_reduction_group) -> None:
    """Set the amax reduction group on a quantizer; no-op if it doesn't support it.

    Unwraps ``DebugQuantizer`` to its ``parent_quantizer``, which is the one that
    actually performs the quantization (and thus the amax reduction).
    """
    if quantizer is None:
        return
    # DebugQuantizer delegates quantization to parent_quantizer
    target = getattr(quantizer, "parent_quantizer", quantizer)
    if target is not None and hasattr(target, "with_amax_reduction"):
        target.with_amax_reduction = amax_reduction_group is not None
        target.amax_reduction_group = amax_reduction_group


def set_quantizer_usage_for_wgrad_all_gather(quantizer) -> None:
    """Configure an all-gather output for consumption by wgrad."""
    if quantizer is None:
        return

    parent_quantizer = getattr(quantizer, "parent_quantizer", None)
    target = parent_quantizer if parent_quantizer is not None else quantizer

    # Hybrid currently gathers in high precision, then quantizes the full
    # result, so request the columnwise representation consumed by wgrad.
    if isinstance(target, HybridQuantizer):
        rowwise_usage, columnwise_usage = False, True
    elif quantizer.supports_only_rowwise_all_gather():
        # Per-tensor FP8 gathers rowwise data and synthesizes its transpose.
        rowwise_usage, columnwise_usage = True, False
    else:
        rowwise_usage, columnwise_usage = False, True

    # Preserve wrapper-specific bookkeeping. In particular, DebugQuantizer
    # propagates usage to its parent while keeping its own state synchronized.
    quantizer.set_usage(rowwise=rowwise_usage, columnwise=columnwise_usage)


def can_reconstruct_wgrad_input_from_original(quantizer) -> bool:
    """Whether wgrad input can be reconstructed from a saved original tensor."""
    target = getattr(quantizer, "parent_quantizer", quantizer)
    if target is None:
        target = quantizer
    if isinstance(target, HybridQuantizer):
        if target.columnwise_source == "original":
            return True
        return target.rowwise_quantizer.is_requantization_safe()
    return target.is_requantization_safe()


def _get_normalization_func(normalization: str, forward: bool):
    fwd_normalization_funcs = {
        "LayerNorm": tex.layernorm_fwd,
        "RMSNorm": tex.rmsnorm_fwd,
    }
    bwd_normalization_funcs = {
        "LayerNorm": tex.layernorm_bwd,
        "RMSNorm": tex.rmsnorm_bwd,
    }

    if forward:
        return fwd_normalization_funcs[normalization]
    return bwd_normalization_funcs[normalization]


def apply_normalization(
    inputmat: torch.Tensor,
    ln_out: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: Union[torch.Tensor, None],
    eps: float,
    output_quantizer,
    output_dtype,
    normalization: str,
    fwd_ln_sm_margin: int,
    zero_centered_gamma: bool,
):
    """Apply normalization to input."""
    normalization_func = _get_normalization_func(normalization, True)

    inputs = (inputmat, ln_weight) if ln_bias is None else (inputmat, ln_weight, ln_bias)

    return normalization_func(
        *inputs,
        eps,
        ln_out,
        output_quantizer,
        TE_DType[output_dtype] if output_dtype in TE_DType else output_dtype,
        fwd_ln_sm_margin,
        zero_centered_gamma,
    )


class _NoopCatFunc(torch.autograd.Function):
    """Concatenate tensors, doing a no-op if possible

    See _noop_cat.

    """

    @staticmethod
    def forward(
        ctx: Any,
        dim: int,
        *tensors: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Check first tensor
        if not tensors:
            raise ValueError("Attempted to concatenate 0 tensors")

        # Check concat dim
        num_dims = tensors[0].dim()
        if not -num_dims <= dim < num_dims:
            raise ValueError(
                "Attempted to concatenate tensor "
                f"with shape {list(tensors[0].size())} along dim {dim}"
            )
        dim %= num_dims

        # Check remaining tensors
        out_shape = list(tensors[0].size())
        split_ranges = [(0, tensors[0].size(dim))]
        for tensor in tensors[1:]:
            in_shape = list(tensor.size())
            if (
                len(in_shape) != num_dims
                or in_shape[:dim] != out_shape[:dim]
                or in_shape[dim + 1 :] != out_shape[dim + 1 :]
            ):
                raise ValueError(
                    "Attempted to concatenate tensors with shapes "
                    f"{[list(tensor.size()) for tensor in tensors]} "
                    f"along dim {dim}"
                )
            split_start = out_shape[dim]
            split_end = split_start + in_shape[dim]
            out_shape[dim] = split_end
            split_ranges.append((split_start, split_end))

        # Save state for backward
        ctx.dim = dim
        ctx.split_ranges = split_ranges

        # Tensor properties from first tensor
        dtype = tensors[0].dtype
        device = tensors[0].device
        strides = tensors[0].stride()
        data_ptr_stride = strides[dim] * tensors[0].element_size()

        # Out-of-place concatenation when view tensors have different storage
        # Note: This works around an edge case with the split_quantize
        # function, which might allocate a buffer and construct
        # subviews. However, in order to reduce CPU overheads, these
        # views are configured manually outside of PyTorch. PyTorch
        # doesn't know these views share the same memory, and it
        # blocks us from reconstructing the full tensor because it
        # thinks we are accessing out-of-bounds memory.
        if tensors[0].untyped_storage().nbytes() < out_shape[dim] * data_ptr_stride:
            return torch.cat(tensors, dim=dim)

        # Out-of-place concatenation if tensor properties do not match
        data_ptr = tensors[0].data_ptr() + tensors[0].size(dim) * data_ptr_stride
        for tensor in tensors[1:]:
            if (
                tensor.dtype != dtype
                or tensor.device != device
                or tensor.stride() != strides
                or tensor.data_ptr() != data_ptr
            ):
                return torch.cat(tensors, dim=dim)
            data_ptr += tensor.size(dim) * data_ptr_stride

        # No-op concatenation
        out = tensors[0].as_strided(out_shape, strides)
        out.requires_grad = any(tensor.requires_grad for tensor in tensors)
        return out

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        grad_inputs = []
        for split_start, split_end in ctx.split_ranges:
            slices = [slice(None)] * grad_output.dim()
            slices[ctx.dim] = slice(split_start, split_end)
            grad_inputs.append(grad_output[tuple(slices)])
        return None, *grad_inputs


def noop_cat(
    tensors: List[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    """Concatenate tensors, doing a no-op if possible

    If tensors are already concatenated in memory, a tensor view of
    that memory region will be returned. Otherwise the tensors will be
    concatenated out-of-place, as usual.

    """
    if not tensors:
        raise ValueError("Attempted to concatenate 0 tensors")
    if len(tensors) == 1:
        return tensors[0]
    if is_in_onnx_export_mode():
        return torch.cat(tensors, dim=dim)
    return _NoopCatFunc.apply(dim, *tensors)


@dataclasses.dataclass
class _ParameterInitMeta:
    """
    Stores essential metadata needed to support deferred parameter initialization.
    """

    init_fn: Optional[Callable] = get_default_init_method()
    get_rng_state_tracker: Optional[Callable] = None
    fp8_meta_index: Optional[int] = None

    def __post_init__(self):
        """Safeguard reference to the parameter's parent module and initialization function."""
        if self.init_fn is None:
            self.init_fn = get_default_init_method()


class WeightGradStream:
    """Runs deferred weight-gradient GEMMs on a side CUDA stream at once.

    The stream mode of :class:`WeightGradStore` (see :meth:`WeightGradStore.set_wgrad_stream`):
    a module whose store is attached to a ``WeightGradStream`` still builds its wgrad closure in
    the backward pass, but ``put()`` launches it immediately on :attr:`stream`, ordered after
    everything enqueued so far on the calling (producer) stream, instead of queueing it for a
    later ``backward_dw()``. The backward pass continues without waiting for it. One
    ``WeightGradStream`` is shared by many modules; its work is serialized on one stream.

    Ordering and lifetime (no ``record_stream``):

    * ``issue()``: an event recorded on the producer stream, waited on by :attr:`stream`, then
      the closure runs with :attr:`stream` current and a completion event is recorded.
    * The tensors and the closure of an issued wgrad stay referenced here until the producer
      stream has waited on its completion. Issued wgrads are grouped (a store attached with
      ``new_group=True`` opens a new group, e.g. at the first wgrad of every layer's backward);
      when a group opens, groups older than ``keep_alive_groups`` are joined into the producer
      stream and released, so the caching allocator recycles their blocks only after a real
      stream dependency. :meth:`join` joins and releases everything (end of the backward).
    * A store attached with ``hold=True`` does not issue at ``put()``: its wgrad is issued at
      the next ``issue()`` (or :meth:`flush` / :meth:`join`), i.e. one producer step later.
    * GEMMs on :attr:`stream` use their own cuBLAS / grouped-GEMM workspaces
      (:func:`register_cublas_workspace_stream`).
    * All of this is CUDA-graph-capture safe: events are pre-created and only recorded /
      waited on, :attr:`stream` forks from and (via :meth:`join`) rejoins the capturing stream.

    Consumers of the weight gradients (e.g. a gradient reduce-scatter) must wait on
    :attr:`stream`; :meth:`join` must run before the end of the backward pass (and before the
    end of a CUDA-graph capture).

    Parameters
    ----------
    stream : torch.cuda.Stream, optional
        Stream to run the wgrads on. Default: a new stream with the lowest priority.
    keep_alive_groups : int, default = 2
        Number of completed groups (besides the current one) whose inputs stay alive without
        a producer-side wait.
    num_events : int, default = 4096
        Size of the event ring (two events per issued wgrad, reused round-robin; must exceed
        the events used between two :meth:`join` calls).
    """

    def __init__(
        self,
        stream: Optional[torch.cuda.Stream] = None,
        *,
        keep_alive_groups: int = 2,
        num_events: int = 4096,
    ):
        if keep_alive_groups < 0:
            raise ValueError(f"keep_alive_groups must be >= 0, got {keep_alive_groups}")
        if stream is None:
            # torch's priority 0 is the default and lowest stream priority.
            stream = torch.cuda.Stream(priority=0)
        self.stream = stream
        self.keep_alive_groups = keep_alive_groups
        # pylint: disable=import-outside-toplevel
        from ..cpp_extensions.gemm import register_cublas_workspace_stream

        register_cublas_workspace_stream(stream)
        self._events = [torch.cuda.Event() for _ in range(num_events)]
        self._event_idx = 0
        self._events_since_join = 0
        # Issued and not yet released: deque of [bundles, last completion event].
        self._groups = collections.deque()
        self._held = None
        self._post_issue_hooks = []
        self.stats = collections.Counter()

    def _next_event(self) -> torch.cuda.Event:
        event = self._events[self._event_idx]
        self._event_idx = (self._event_idx + 1) % len(self._events)
        self._events_since_join += 1
        if self._events_since_join > len(self._events):
            raise RuntimeError(
                "WeightGradStream: event ring exhausted between two join() calls; "
                "increase num_events."
            )
        return event

    def add_post_issue_hook(self, hook: Callable[[Optional[str]], None]) -> None:
        """Call ``hook(tag)`` after every issued wgrad, with :attr:`stream` current (e.g. to
        interleave further work on the stream)."""
        self._post_issue_hooks.append(hook)

    def _launch(self, tensor_list, func, tag, producer) -> None:
        ready = self._next_event()
        ready.record(producer)
        self.stream.wait_event(ready)
        with torch.cuda.stream(self.stream):
            func(*tensor_list)
            for hook in self._post_issue_hooks:
                hook(tag)
        done = self._next_event()
        done.record(self.stream)
        if not self._groups:
            self._groups.append([[], None])
        group = self._groups[-1]
        group[0].append((tensor_list, func))
        group[1] = done
        self.stats["issued"] += 1
        self.stats[f"issued_{tag}"] += 1
        if torch.cuda.is_current_stream_capturing():
            self.stats["issued_captured"] += 1

    def _release_older_groups(self, producer, keep: int) -> None:
        while len(self._groups) > keep + 1:
            _, done = self._groups.popleft()
            if done is not None:
                # The stream is in order: its last completion event covers the whole group.
                producer.wait_event(done)
            self.stats["groups_released"] += 1

    def issue(
        self,
        tensor_list: List[Any],
        func: Callable,
        *,
        tag: Optional[str] = None,
        hold: bool = False,
        new_group: bool = False,
    ) -> None:
        """Run ``func(*tensor_list)`` on :attr:`stream` after the work enqueued so far on the
        current stream (see the class docstring for ``hold`` / ``new_group``)."""
        producer = torch.cuda.current_stream()
        if producer.cuda_stream == self.stream.cuda_stream:
            # Already on the wgrad stream (nested use): stream order is enough.
            func(*tensor_list)
            return
        if new_group:
            self._groups.append([[], None])
            self._release_older_groups(producer, self.keep_alive_groups)
        if self._held is not None:
            held, self._held = self._held, None
            self._launch(*held, producer)
        if hold:
            self._held = (tensor_list, func, tag)
            self.stats["held"] += 1
            return
        self._launch(tensor_list, func, tag, producer)

    def flush(self) -> None:
        """Issue a held wgrad now (ordered after the current stream)."""
        if self._held is not None:
            held, self._held = self._held, None
            self._launch(*held, torch.cuda.current_stream())

    def has_pending(self) -> bool:
        """Whether wgrads were issued (or held) since the last :meth:`join`."""
        return self._held is not None or any(group[1] is not None for group in self._groups)

    def join(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Flush, make ``stream`` (default: the current stream) wait for everything issued on
        :attr:`stream` and release all kept-alive inputs."""
        if stream is None:
            stream = torch.cuda.current_stream()
        if self._held is not None:
            with torch.cuda.stream(stream):
                self.flush()
        if any(group[1] is not None for group in self._groups):
            stream.wait_stream(self.stream)
            self.stats["joins"] += 1
        self._groups.clear()
        self._events_since_join = 0


class WeightGradStore:
    """
    A class to manage weight gradient storage and computation in Transformer modules.
    This class enables split backward propagation for better memory efficiency.

    Stream mode (:meth:`set_wgrad_stream`): ``put()`` runs the wgrad at once on the stream
    of a :class:`WeightGradStream` instead of queueing it for ``backward_dw()``.
    """

    def __init__(self, delay_wgrad_compute=False, ub_bulk_wgrad=False):
        """
        Initialize the WeightGradStore.

        Args:
            delay_wgrad_compute (bool): Whether to delay weight gradient computation
            ub_bulk_wgrad (bool): Whether to enable bulk weight gradient computation
        """
        if delay_wgrad_compute:
            self.context = queue.Queue()
            assert (
                ub_bulk_wgrad is False
            ), "ub_bulk_wgrad is not supported when enabling delay_wgrad_compute"
            self.enabled = delay_wgrad_compute
        else:
            self.context = None
            self.enabled = False
        self.ub_bulk_wgrad = ub_bulk_wgrad
        self.wgrad_stream = None
        self.wgrad_stream_kwargs = {}

    def set_wgrad_stream(
        self,
        wgrad_stream: Optional[WeightGradStream],
        *,
        tag: Optional[str] = None,
        hold: bool = False,
        new_group: bool = False,
    ) -> None:
        """Attach (or with ``None`` detach) a :class:`WeightGradStream`.

        While attached, :meth:`delay_wgrad_compute` is True and :meth:`put` issues the wgrad
        on the stream at once (``tag`` / ``hold`` / ``new_group``: see
        :meth:`WeightGradStream.issue`); there is nothing for ``backward_dw()`` to run. The
        wgrad must write its result in place (fused accumulation into ``main_grad``): a
        module checks that before attaching. Not combinable with ``delay_wgrad_compute=True``
        (``backward_dw()`` deferral) or bulk Userbuffers wgrad.
        """
        if wgrad_stream is not None:
            if self.enabled:
                raise ValueError(
                    "set_wgrad_stream: the store already defers wgrads to backward_dw() "
                    "(delay_wgrad_compute=True)"
                )
            if self.ub_bulk_wgrad:
                raise ValueError("set_wgrad_stream: not supported with ub_bulk_wgrad")
        self.wgrad_stream = wgrad_stream
        self.wgrad_stream_kwargs = {"tag": tag, "hold": hold, "new_group": new_group}

    def stream_mode(self) -> bool:
        """Whether wgrads are issued on a :class:`WeightGradStream` at ``put()``."""
        return self.wgrad_stream is not None

    def delay_wgrad_compute(self):
        """
        Get the current split backward propagation status.

        Returns:
            bool: True if split backward is enabled (or the store is in stream mode),
            False otherwise
        """
        return self.enabled or self.wgrad_stream is not None

    def defer_wgrad(self, produces_bias_grad: bool = False) -> bool:
        """Whether a module's backward should hand its wgrad closure to :meth:`put`.

        ``produces_bias_grad``: the closure's GEMM also returns the bias gradient (BF16 bias
        epilogue). ``backward_dw()`` assigns that gradient later; in stream mode nothing would,
        so such a wgrad runs inline."""
        if self.wgrad_stream is not None:
            return not produces_bias_grad
        return self.enabled

    def needs_backward_dw(self) -> bool:
        """Whether ``backward_dw()`` has work to run (False in stream mode)."""
        return self.enabled and self.wgrad_stream is None

    def enable_delay_wgrad_compute(self):
        """Enable split backward propagation."""
        self.enabled = True

    def disable_delay_wgrad_compute(self):
        """Disable split backward propagation."""
        self.enabled = False

    def put(self, tensor_list, func):
        """
        Store tensors and computation function for later execution, or in stream mode
        run the computation now on the wgrad stream.

        Args:
            tensor_list (list): List of tensors needed for computation
            func (callable): Function to be executed with the tensors
        """
        if self.wgrad_stream is not None:
            self.wgrad_stream.issue(tensor_list, func, **self.wgrad_stream_kwargs)
            return
        assert self.enabled is True, "delay_wgrad_compute is not enabled"
        self.context.put([tensor_list, func])

    def pop(self):
        """
        Execute the stored computation with the stored tensors.
        Raises an exception if the queue is empty.
        """
        assert self.wgrad_stream is None, "WeightGradStore in stream mode has nothing to pop"
        assert self.enabled is True, "delay_wgrad_compute is not enabled"
        if self.context.qsize() > 0:
            tensor_list, func = self.context.get()
            return func(*tensor_list), tensor_list
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            raise RuntimeError(f"Pop empty queue. rank {rank}")
        raise RuntimeError("Pop empty queue. No distributed environment detected.")

    def assert_empty(self):
        """
        Assert that the queue is empty.
        Used for debugging and ensuring proper cleanup.
        """
        if self.wgrad_stream is not None:
            return
        assert self.enabled is True, "delay_wgrad_compute is not enabled"
        rank = torch.distributed.get_rank()
        assert self.context.empty(), f"Queue is not empty. rank {rank}"


def check_fp8_reduce_and_update(restore_first_module: bool = False) -> bool:
    """Whether this module's backward should reduce and update the FP8 scaling factors.

    Consumes the "first FP8 module" flag, restored when the forward is a
    recomputation so the flag survives for the real forward's owner.
    """
    qstate = FP8GlobalStateManager.quantization_state
    first_fp8_module = qstate.is_first_fp8_module
    result = FP8GlobalStateManager.is_first_fp8_module()
    if restore_first_module or in_fp8_activation_recompute_phase():
        qstate.is_first_fp8_module = first_fp8_module
    return result
