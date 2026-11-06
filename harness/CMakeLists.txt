cmake_minimum_required(VERSION 3.18)

find_package(OpenCV QUIET)

add_executable(stitch_harness main.cpp pipeline_runner.cu)
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

add_executable(stress_multires_cuda stress_multires.cu pipeline_runner.cu)
target_link_libraries(stress_multires_cuda PRIVATE stitch_common stitch_controller stitch_s1 stitch_s2_s3 stitch_s4 stitch_s5 stitch_s6)
target_include_directories(stress_multires_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set_target_properties(stress_multires_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)

add_executable(test_adaptive_upload ../tests/controller/test_adaptive_upload.cu pipeline_runner.cu)
target_link_libraries(test_adaptive_upload PRIVATE
  stitch_common stitch_controller stitch_s1 stitch_s2_s3 stitch_s4 stitch_s5 stitch_s6)
target_include_directories(test_adaptive_upload PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../tests)
add_test(NAME test_adaptive_upload COMMAND test_adaptive_upload)
set_tests_properties(test_adaptive_upload PROPERTIES SKIP_RETURN_CODE 77)
