cmake_minimum_required(VERSION 3.20.0)

set(VARBOOL TRUE)

# if
if(VARBOOL)
  message(TRUE)
else()
  message(FALSE)
endif()


# NOT
if(NOT VARBOOL)
  message(TRUE)
else()
  message(FALSE)
endif()


# OR
if(NOT VARBOOL OR VARBOOL)
  message(TRUE)
else()
  message(FALSE)
endif()


# AND
if(NOT VARBOOL AND VARBOOL)
  message(TRUE)
else()
  message(FALSE)
endif()


if(1 LESS 2)
  message("1 < 2")
endif()

# 注意，字母字符串与数字最好不要比较，否则会比较不成功。Cmake会首先尝试将字符串转换为数字
if("o" LESS 200)
  message("o < 200")
endif()
if("o" GREATER_EQUAL 200)
  message("o > 200")
endif()
if("2" EQUAL 2)
  message("2==2")
endif()
message("After the if statement")




# for 推荐使用for不用while
foreach(VAR RANGE 3)
  message(${VAR})
endforeach()

message("------------------------------------")
set(MY_LIST 1 2 3)
# 用的比较多，因为常常增加一些东西，用完就扔了
foreach(VAR IN LISTS MY_LIST ITEMS 4 f)
  message(${VAR})
endforeach()


# zip操作
message("---------------------")
set(L1 one two three four)
set(L2 1 2 3 4 5)

foreach(num IN ZIP_LISTS L1 L2)
    message("word = ${num_0},num = ${num_1}")
endforeach()
