cmake_minimum_required(VERSION 3.0)

project(jnn LANGUAGES C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pedantic")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

include_directories(libjnn/include/)

add_subdirectory(bin)
add_subdirectory(libjnn)
