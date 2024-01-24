cmake_minimum_required(VERSION 3.20.0)

function(MyFunc FirstArg)
  message("MYFunc Name: ${CMAKE_CURRENT_FUNCTION}")
  message("FirstArg = ${FirstArg}")
  set(FirstArg "New value")
  message("FirstArg After change = ${FirstArg}")
  # 另一种打印参数的方式固有变量 ARGVn
  message("ARGV0 ${ARGV0}")
  message("ARGV1 ${ARGV1}")
  message("ARGV2 ${ARGV2}")
endfunction()


set(FirstArg "first value")
MyFunc(${FirstArg} "value")
# FirstArg没有改变代表函数的FirstArg只在函数内的作用域
message("FirstArg After Function =  ${FirstArg}")