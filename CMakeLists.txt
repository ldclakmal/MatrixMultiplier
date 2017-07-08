# cmake_minimum_required(VERSION <specify CMake version here>)
project(Project)

set(CMAKE_CXX_STANDARD 11)

# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(SOURCE_FILES sequential_matrix_multiplier.cpp)
add_executable(Project ${SOURCE_FILES})