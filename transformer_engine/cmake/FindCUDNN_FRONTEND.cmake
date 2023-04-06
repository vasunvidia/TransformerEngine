add_library(CUDNN_FRONTEND::cudnn_frontend INTERFACE IMPORTED)

set(CMAKE_FIND_DEBUG_MODE TRUE)
find_path(
    CUDNN_FRONTEND_INCLUDE_DIR cudnn_frontend.h
    #HINTS $ENV{CUDNN_FRONTEND_PATH} ${CUDNN_PATH} ${CUDAToolkit_INCLUDE_DIRS}
    HINTS $ENV{CUDNN_FRONTEND_PATH} ${CUDNN_FRONTEND_PATH} ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES include
)
set(CMAKE_FIND_DEBUG_MODE FALSE)

message(STATUS "cuDNN_frontend0: $ENV{CUDNN_FRONTEND_PATH}")
message(STATUS "cuDNN_frontend1: ${CUDNN_FRONTEND_INCLUDE_DIR}")
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    LIBRARY REQUIRED_VARS
    CUDNN_FRONTEND_INCLUDE_DIR 
)

if(CUDNN_FRONTEND_INCLUDE_DIR)

    message(STATUS "cuDNN_frontend: ${CUDNN_FRONTEND_INCLUDE_DIR}")
    
    set(CUDNN_FRONTEND_FOUND ON CACHE INTERNAL "cuDNN_frontend Library Found")

else()

    set(CUDNN_FRONTEND_FOUND OFF CACHE INTERNAL "cuDNN_frontend Library Not Found")

endif()

target_include_directories(
    CUDNN_FRONTEND::cudnn_frontend
    INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CUDNN_FRONTEND_INCLUDE_DIR}>
)

