
cmake_minimum_required(VERSION 3.20.0)


set(Var1 "YZZY")
message(${Var1})

set("My Var" zzz)
message(${My\ Var})

# 存储多个值
set(LISTVALUE a1 a2)
set(LISTVALUE a1;a2)#一个意思
message(${LISTVALUE})

# 打印环境变量
message($ENV{PATH})# 这里的PATH是ENV环境中的变量

# 增加环境中的变量，作用域是这个cmake文件
set(ENV{CXX} "g++") #创建CXX变量值为g++
message($ENV{CXX})


# unset
unset(ENV{CXX})
message($ENV{CXX})# 因为没有设置，所以会报错