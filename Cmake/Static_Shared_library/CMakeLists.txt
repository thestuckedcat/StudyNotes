cmake_minimum_required(VERSION 3.20.0)
# 该CMakeLists主要是生成库
project(Animal CXX)

# add_subdirectory(animal)

# 不是生成可执行文件
# add_executable(Animal main.cpp)

# target_link_libraries(Animal PUBLIC AnimalLib)
# message("PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR}")
# target_include_directories(Animal PUBLIC "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/animal")

# 生成静态库
# 将所有Cpp文件加入SRC变量
file(GLOB SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)

include_directories((${PROJECT_SOURCE_DIR}/include))

# 设置输出的library的路径（根目录下的a文件夹）
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/a)
add_library(animal_static STATIC ${SRC})


# 生成动态库

# file(GLOB SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)

# include_directories((${PROJECT_SOURCE_DIR}/include))

# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/so)
# add_library(animal_shared SHARED ${SRC})

