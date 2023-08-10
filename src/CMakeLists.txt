set(CMAKE_BINARY_DIR "${PROJECT_BINARY_DIR}/lib")

include_directories(../include)
add_library(llawa STATIC llawa.c)