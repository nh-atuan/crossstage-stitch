cmake_minimum_required(VERSION 3.18)

find_package(OpenCV QUIET)

add_executable(stitch_harness main.cpp pipeline_runner.cu)
target_link_libraries(stitch_harness PRIVATE
  stitch_common
  stitch_s1
  stitch_s2_s3
  stitch_s4
  stitch_s5
  stitch_s6
)

if(OpenCV_FOUND)
  target_compile_definitions(stitch_harness PRIVATE STITCH_HAS_OPENCV)
  target_link_libraries(stitch_harness PRIVATE ${OpenCV_LIBS})
  target_include_directories(stitch_harness PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

target_include_directories(stitch_harness PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

set_target_properties(stitch_harness PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED YES
)
