cmake_minimum_required(VERSION 3.20.0)

project(Animal CXX)

include(animal/animal.cmake)
message(${animal_sources})
add_executable(Animal
main.cpp
${animal_sources}
)

