cmake_minimum_required(VERSION 3.20.0)

# 两种方式创建Var
set(LISTVALUE a1 a2 a3)
message(${LISTVALUE})

list(APPEND port p1 p2 p3)
message(${port})

# 获取长度 (标识符 目标list 输出变量)
list(LENGTH LISTVALUE len)
message(${len})

# 查找(标识符 目标list 索要查找的内容 结果index)，没找到返回-1
list(FIND LISTVALUE a4 index)
message(${index})

# 删除
list(REMOVE_ITEM port p1)
message(${port})

# 添加
list(APPEND LISTVALUE a5)
message(${LISTVALUE})

# 插入 ，(标识符 目标list 插入index 插入的目标)注意，插入会发生在index这个位置上，即为插入到index这个位置旧的元素之前
list(INSERT LISTVALUE 3 a4)
message(${LISTVALUE})

# 翻转
list(REVERSE LISTVALUE)
message(${LISTVALUE})

# 排序（字典序
list(SORT LISTVALUE)
message(${LISTVALUE})
