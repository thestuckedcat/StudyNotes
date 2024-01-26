cmake_minimum_required(VERSION 3.20.0)

project(Animal CXX)

# 引入头文件
# include_directories(${PROJECT_SOURCE_DIR}/include)


# 添加静态库目录
# link_directories(${PROJECT_SOURCE_DIR}/a)
# 连接静态库
# link_libraries(animal_static)


# add_executable(app main.cpp)



include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_SOURCE_DIR}/so)
add_executable(app main.cpp)
target_link_libraries(app PUBLIC animal_shared)




