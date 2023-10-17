cmake_minimum_required(VERSION 3.5.1)
project(mnist_study VERSION 0.0.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA
find_package(CUDA)
# TensorRT
find_library(NVINFER NAMES nvinfer)
find_library(NVONNXPARSER NAMES nvonnxparser)
find_library(CUDART NAMES cudart PATHS /usr/local/cuda/lib64)

include_directories(utils ${CUDA_INCLUDE_DIRS})

# utils so
add_library(utils SHARED utils/mnist-reader.cc)

# devicesInfo
add_executable(deviceInfo deviceInfo.cpp)
target_link_libraries(deviceInfo
    ${NVINFER}
    ${NVONNXPARSER}
    ${CUDART}
    utils)

# FP16
add_executable(onnx2engine.fp16 fp16/onnx2engine.fp16.cpp)
target_link_libraries(onnx2engine.fp16
    ${NVINFER}
    ${NVONNXPARSER}
    ${CUDART}
    utils)
add_executable(infer.fp16 fp16/infer.fp16.cpp)
target_link_libraries(infer.fp16
    ${NVINFER}
    ${NVONNXPARSER}
    ${CUDART}
    utils)

set(CUDA_VERBOSE_BUILD ON)
message(STATUS "CUDA_VERSION_MAJOR: ${CUDA_VERSION_MAJOR}")
message(STATUS "CUDA_VERSION_MINOR: ${CUDA_VERSION_MINOR}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
message(STATUS "CUDA_VERSION_STRING: ${CUDA_VERSION_STRING}")
message(STATUS "CUDA_HAS_FP16: ${CUDA_HAS_FP16}")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA_SDK_ROOT_DIR: ${CUDA_SDK_ROOT_DIR}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_CUFFT_LIBRARIES: ${CUDA_CUFFT_LIBRARIES}")
message(STATUS "CUDA_CUBLAS_LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")
