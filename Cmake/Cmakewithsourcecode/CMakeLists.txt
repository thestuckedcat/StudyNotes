cmake_minimum_required(VERSION 3.20.0)

# 设置c++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# config
configure_file(config.h.in config.h)

project(Animal CXX)

add_subdirectory(animal)

add_executable(Animal
main.cpp

)

target_link_libraries(Animal PUBLIC AnimalLib)
message("PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR}")

# 这里包含了animal的头文件目录以及config.h.in生成的config.h的目录PROJECT_BINARY_DIR
target_include_directories(Animal PUBLIC "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/animal")



