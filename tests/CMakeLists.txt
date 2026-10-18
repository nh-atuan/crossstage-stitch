add_executable(test_common test_common.cpp)
target_link_libraries(test_common PRIVATE stitch_common)
target_include_directories(test_common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_common COMMAND test_common)

# Compile the shared headers with nvcc (sm_75) when CUDA is available.
if(CMAKE_CUDA_COMPILER)
  add_executable(test_common_cuda test_common_cuda.cu)
  target_link_libraries(test_common_cuda PRIVATE stitch_common)
  add_test(NAME test_common_cuda COMMAND test_common_cuda)
endif()
