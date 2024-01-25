cmake_minimum_required(VERSION 3.20.0)
# 宏名 宏参数
macro(Test myVar)
  set(myVar "new value") # 这里是创建了一个新的myVar变量
  message("argument: ${myVar}")
endmacro()

set(myVar "First value")
message("myVar: ${myVar}")
Test("value")
message("myVar: ${myVar}")