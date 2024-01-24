cmake_minimum_required(VERSION 3.20)

message("hello")

message("这是一个
换行")

message([[这也是
一个
换行]])

# 获取信息 ${}
# 通常使用${}来获取一个变量的值，例如获取Cmake版本
message(${CMAKE_VERSION})