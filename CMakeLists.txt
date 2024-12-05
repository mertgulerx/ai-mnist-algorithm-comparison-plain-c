cmake_minimum_required(VERSION 3.30)
project(mnist-algorithm-comparison-plain-c C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(mnist-algorithm-comparison-plain-c main.c)
