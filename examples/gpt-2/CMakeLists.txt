set(CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${PROJECT_BINARY_DIR}/bin")
message("Executable files will at ${CMAKE_BINARY_DIR}")

include_directories(${LLAWA_INCLUDE})
message(${LLAWA_INCLUDE})

add_executable(gpt2 gpt2.cpp)
target_link_libraries(gpt2 llawa)
