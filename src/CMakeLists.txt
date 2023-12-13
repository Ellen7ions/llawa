set(CMAKE_BINARY_DIR "${PROJECT_BINARY_DIR}/lib")

include_directories(../include)
add_library(llawa STATIC llawa.c
)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(llawa PUBLIC OpenMP::OpenMP_CXX)
endif()