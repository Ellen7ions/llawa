set(CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests")
#message("Tests files will at ${CMAKE_BINARY_DIR}")

include_directories(${LLAWA_INCLUDE})
message(${LLAWA_INCLUDE})

add_executable(test_ops test_ops.cpp)
target_link_libraries(test_ops llawa)

add_executable(test_mat_mul test_mat_mul.cpp)
target_link_libraries(test_mat_mul llawa)