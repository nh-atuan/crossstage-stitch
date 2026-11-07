cmake_minimum_required(VERSION 3.18)

find_package(OpenCV QUIET)

add_executable(stitch_harness main.cpp pipeline_runner.cu ../src/controller/features.cu ../src/controller/relational_features.cu ../src/controller/spatial.cu)
if(DEFINED STITCH_SOURCE_REV)
  target_compile_definitions(stitch_harness PRIVATE STITCH_GIT_REV="${STITCH_SOURCE_REV}")
endif()
target_link_libraries(stitch_harness PRIVATE
  stitch_common
  stitch_controller
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

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
  target_include_directories(stitch_harness PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

set_target_properties(stitch_harness PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED YES
)
