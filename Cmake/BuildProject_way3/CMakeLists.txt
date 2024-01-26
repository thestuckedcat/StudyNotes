cmake_minimum_required(VERSION 3.20.0)

project(Animal CXX)

add_subdirectory(animal)

add_executable(Animal
main.cpp

)

target_link_libraries(Animal PUBLIC AnimalLib)
message("PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR}")
target_include_directories(Animal PUBLIC "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/animal")



