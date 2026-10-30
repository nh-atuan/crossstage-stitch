cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(crossstage_controller LANGUAGES CXX)
  enable_testing()
endif()

get_filename_component(STITCH_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
if(NOT TARGET stitch_common)
  add_library(stitch_common INTERFACE)
  target_include_directories(stitch_common INTERFACE "${STITCH_REPO_ROOT}/src")
endif()

add_library(stitch_controller STATIC controller.cpp policy.cpp validation.cpp reuse.cpp)
target_link_libraries(stitch_controller PUBLIC stitch_common)
target_compile_features(stitch_controller PUBLIC cxx_std_17)

add_executable(test_controller "${STITCH_REPO_ROOT}/tests/controller/test_controller.cpp")
target_include_directories(test_controller PRIVATE "${STITCH_REPO_ROOT}/tests")
target_link_libraries(test_controller PRIVATE stitch_controller)
add_test(NAME test_controller COMMAND test_controller)

add_executable(controller_trace "${STITCH_REPO_ROOT}/tests/controller/controller_trace.cpp")
target_link_libraries(controller_trace PRIVATE stitch_controller)
add_test(NAME controller_trace COMMAND controller_trace)
