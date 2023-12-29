set(CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
message("Executable files will at ${CMAKE_BINARY_DIR}")

option(OMP_PERF "turn on debug mode" ON)
set(OMP_PERF ON)

if (${OMP_PERF})
    message("OMP_PERF ON")
    add_definitions(-DOMP_PERF)
endif ()

include_directories(${LLAWA_INCLUDE})
message(${LLAWA_INCLUDE})

find_package(OpenMP REQUIRED)
add_executable(gpt2 gpt2.cpp)

target_link_libraries(gpt2 PRIVATE llawa)
target_link_libraries(gpt2
        PRIVATE
        OpenMP::OpenMP_CXX
)
