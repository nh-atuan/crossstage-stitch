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
  set(CMAKE_CUDA_ARCHITECTURES 75)
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

add_subdirectory(tests)
