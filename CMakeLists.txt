cmake_minimum_required(VERSION 3.23)
project(llawa)

set(LLAWA_SRC "${PROJECT_SOURCE_DIR}/src")
set(LLAWA_INCLUDE "${PROJECT_SOURCE_DIR}/include")


option(LLAWA_DEBUG "turn on debug mode" ON)


if (${LLAWA_DEBUG})
    message("DEBUG ON")
    add_definitions(-DLLAWA_DEBUG)
endif ()

#add_subdirectory(tests)
add_subdirectory(src)
add_subdirectory(examples)

add_subdirectory(tests)