cmake_minimum_required(VERSION 3.18)
project(crossstage_stitch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA is enabled when a toolkit is found (always the case on Colab T4).
# Without it only the host-side targets (headers, tests) are built.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  enable_language(CUDA)
else()
  message(STATUS "CUDA compiler not found: building host-only targets")
endif()

enable_testing()

# Shared interface headers (src/common). Header-only, no CUDA dependency.
add_library(stitch_common INTERFACE)
target_include_directories(stitch_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_subdirectory(src/s1)
add_subdirectory(src/s2_s3)
add_subdirectory(src/s4)
add_subdirectory(src/s5)
add_subdirectory(src/s6)

if(CMAKE_CUDA_COMPILER)
  set(STITCH_CONTROLLER_GPU_TESTS ON)
endif()
add_subdirectory(src/controller)

if(CMAKE_CUDA_COMPILER)
  add_subdirectory(harness)
endif()

add_subdirectory(tests)
