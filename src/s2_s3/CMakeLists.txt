cmake_minimum_required(VERSION 3.18)

# Standalone stage build; shared root build files remain owned by Tuan.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(crossstage_s2_s3 LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  enable_testing()
endif()

get_filename_component(STITCH_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
if(NOT TARGET stitch_common)
  add_library(stitch_common INTERFACE)
  target_include_directories(stitch_common INTERFACE "${STITCH_REPO_ROOT}/src")
endif()

add_library(stitch_s2_s3_config STATIC s2_s3_config.cpp)
target_link_libraries(stitch_s2_s3_config PUBLIC stitch_common)
target_compile_features(stitch_s2_s3_config PUBLIC cxx_std_17)
add_executable(test_s2_s3_config "${STITCH_REPO_ROOT}/tests/test_s2_s3_config.cpp")
target_include_directories(test_s2_s3_config PRIVATE "${STITCH_REPO_ROOT}/tests")
target_link_libraries(test_s2_s3_config PRIVATE stitch_s2_s3_config)
add_test(NAME test_s2_s3_config COMMAND test_s2_s3_config)
