add_executable(test_common test_common.cpp)
target_link_libraries(test_common PRIVATE stitch_common)
target_include_directories(test_common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_common COMMAND test_common)

add_executable(test_harness_config test_harness_config.cpp)
target_link_libraries(test_harness_config PRIVATE stitch_common stitch_s6_geometry)
target_include_directories(test_harness_config PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME test_harness_config COMMAND test_harness_config)

add_executable(test_attempt_trace controller/test_attempt_trace.cpp)
target_link_libraries(test_attempt_trace PRIVATE stitch_common stitch_s6_geometry)
target_include_directories(test_attempt_trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME test_attempt_trace COMMAND test_attempt_trace)

add_executable(test_controller_policy controller/test_policy_loader.cpp)
target_link_libraries(test_controller_policy PRIVATE stitch_common stitch_controller)
target_include_directories(test_controller_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME test_controller_policy COMMAND test_controller_policy)

# Compile the shared headers with nvcc (sm_75) when CUDA is available.
if(CMAKE_CUDA_COMPILER)
  add_executable(test_harness_upload test_harness_upload.cu ../harness/pipeline_runner.cu)
  target_include_directories(test_harness_upload PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(test_harness_upload PRIVATE stitch_common stitch_controller
    stitch_s1 stitch_s2_s3 stitch_s4 stitch_s5 stitch_s6)
  set_target_properties(test_harness_upload PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
  add_test(NAME test_harness_upload COMMAND test_harness_upload)
  set_tests_properties(test_harness_upload PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(test_common_cuda test_common_cuda.cu)
  target_link_libraries(test_common_cuda PRIVATE stitch_common)
  add_test(NAME test_common_cuda COMMAND test_common_cuda)
endif()
