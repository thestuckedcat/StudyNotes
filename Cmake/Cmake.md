## 1. 在linux中编译可执行文件

```bash
g++ main.cpp -o ExecutionFilename
```





```bash
stuckedcat@ubuntu:~$ mkdir cmake_tutorials
stuckedcat@ubuntu:~$ ls
cmake_tutorials  Downloads   ndiff-2.00.tar.gz            Pictures  Templates
Desktop          Music       ParallelComputing            Public    Videos
Documents        ndiff-2.00  ParallelComputingSourceCode  snap
stuckedcat@ubuntu:~$ cd cmake_tutorials
stuckedcat@ubuntu:~/cmake_tutorials$ mkdir 1.moduel1
stuckedcat@ubuntu:~/cmake_tutorials$ cd 1.moduel1/
stuckedcat@ubuntu:~/cmake_tutorials/1.moduel1$ touch main.cpp
```

```c++
#include <iostream>

int main(){
	float first_no, second_no, result_add, result_div;
	std::cout<<"Enter first number\t";
	std::cin >> first_no;
	std::cout << "Enter second number\t";
	std::cin >> second_no;
	
	result_add = first_no + second_no;
	result_div = first_no / second_no;
	
	std::cout << result_add << " " << result_div;

}
```

生成一个名为calculator的可执行文件，并执行。

```bash
stuckedcat@ubuntu:~/cmake_tutorials/1.moduel1$ g++ main.cpp -o calculator
stuckedcat@ubuntu:~/cmake_tutorials/1.moduel1$ ./calculator
Enter first number	1
Enter second number	2
3 0.5
```



多个文件链接

```bash
stuckedcat@ubuntu:~/cmake_tutorials/1.moduel1$ touch addition.cpp division.cpp print_result.cpp
```

注意，

* main中必须declare对应的函数
* 我们必须告诉g++定义的函数在哪里



```c++
//main.cpp
#include <iostream>

float addition( float, float );
float division(float, float);
void print_result( std::string, float);

int main(){

float first_no, second_no, result_add, result_div;

std::cout<< "Enter first number\t";
std::cin>> first_no;
std::cout<< "Enter second number\t";
std::cin>> second_no;

result_add = addition(first_no , second_no);
result_div = division(first_no , second_no);

print_result("Addition", result_add);
print_result("Division", result_div);
//std::cout<< "Addition result:\t"<< result_add<< "\nDivision result:\t"<< result_div<< "\n";

return 0;

}

//addition.cpp
float addition( float num1, float num2 ){
	return num1+num2+0;
}

//division.cpp
float division(float num1, float num2){
	return num1/num2+0;
}

//print_result.cpp
#include <iostream>

void print_result( std::string result_type, float result_value){
	std::cout<< result_type<< " result:\t"<< result_value<< "\n";
}

```



![image-20240123204455717](./assets/image-20240123204455717.png)

更推荐的方法其实是将declaration写在头文件中，然后包含这些头文件。

==头文件的作用就是将declaration复制到main.cpp中。==

最终我们会得到

```c++
// addition.h
float addition( float, float );
// division.h
float division(float, float);
// print_result.h
void print_result( std::string, float);

//main.cpp
#include <iostream>
#include "addition.h"
#include "division.h"
#include "print_result.h"
int main(){

float first_no, second_no, result_add, result_div;

std::cout<< "Enter first number\t";
std::cin>> first_no;
std::cout<< "Enter second number\t";
std::cin>> second_no;

result_add = addition(first_no , second_no);
result_div = division(first_no , second_no);

print_result("Addition", result_add);
print_result("Division", result_div);
//std::cout<< "Addition result:\t"<< result_add<< "\nDivision result:\t"<< result_div<< "\n";

return 0;

}

//addition.cpp
float addition( float num1, float num2 ){
	return num1+num2+0;
}

//division.cpp
float division(float num1, float num2){
	return num1/num2+0;
}

//print_result.cpp
#include <iostream>

void print_result( std::string result_type, float result_value){
	std::cout<< result_type<< " result:\t"<< result_value<< "\n";
}

```





## 2. 编译原理

最初，所有我呢见都是相互独立编译的。

![image-20240123205035689](./assets/image-20240123205035689.png)

因为main.cpp使用了其他文件中的函数，因此我们必须告诉编译器，main.cpp中的三个函数确实是存在某处的。

我们可以通过直接在使用到这些函数的文件中声明这些函数来做到这一点，我们也可以通过包含头文件来做到这一点。

头文件的作用主要是打包，实际上编译器会将头文件的声明复制到main.cpp中。





在此时，==编译器在编译阶段并不关心这些函数的定义==。编译器在编译过程中会在call function的地方放一个占位符，这些占位符告诉函数调用会在**链接**阶段得到解决。

==链接阶段==，链接器找到`addition.cpp,division.cpp,print_result.cpp`的编译二进制文件，并且将他们链接到一起生成一个可执行文件。



这里需要注意的一点是，此时你拥有

* 拥有函数声明的头文件
* .cpp文件被编译后的二进制文件。

你才能够构建该项目。

<img src="./assets/image-20240123205815932.png" alt="image-20240123205815932" style="zoom: 50%;" />





## 3.Makefile初探

通常为了构建项目，开发人员会编写makefile专门构建系统并链接源代码。

```bash
sudo apt install make
touch makefile
```

![image-20240123210507137](./assets/image-20240123210507137.png)

![image-20240123210542434](./assets/image-20240123210542434.png)

make命令将查找makefile然后根据makefile的内容构建项目。

执行后的输出意味着编译器编译了所有文件，然后将它们链接到了一起。

然后，我们就可以使用可执行文件calculator来运行程序了。





我们可以修改addition.cpp然后重新make

![image-20240123210817930](./assets/image-20240123210817930.png)

可以发现，现在只有被修改的文件被重新编译了。这在大型文件系统节省了我们的时间。

> 让我们使用五个文件来更加清晰地描述编译过程中的依赖关系和编译步骤：
>
> 1. **main.cpp**：这是包含 `main` 函数的源文件，假设它调用了 `function1.h` 和 `function2.h` 中声明的函数。
> 2. **function1.h**：这个头文件声明了 `function1.cpp` 中定义的函数。
> 3. **function1.cpp**：这个源文件包含 `function1.h` 中声明的函数的定义。
> 4. **function2.h**：这个头文件声明了 `function2.cpp` 中定义的函数。
> 5. **function2.cpp**：这个源文件包含 `function2.h` 中声明的函数的定义。
>
> 根据这些文件，我们来描述编译过程中的三种情况：
>
> 1. **编译项目第一次**：
>    - 你需要所有五个文件：`main.cpp`, `function1.h`, `function1.cpp`, `function2.h`, `function2.cpp`。
>    - 编译器需要头文件来了解函数的声明，需要源文件（.cpp）来获取函数的实现。
>    - 编译这些文件会生成对象文件（`.o` 或 `.obj`），然后这些对象文件被链接成最终的可执行文件。
> 2. **第二次编译时，如果头文件和源文件没有变化**：
>    - 你只需要已经编译的对象文件和任何改变的源文件。
>    - 如果 `main.cpp` 没有改变，并且 `function1.cpp` 和 `function2.cpp` 也没有改变，则无需重新编译，直接链接已有的对象文件即可生成可执行文件。
>    - 但如果 `main.cpp` 发生了改变，只需重新编译 `main.cpp`，然后用新生成的对象文件与其他旧的对象文件链接。
> 3. **第二次编译时，如果某个源文件（如 `function1.cpp`）发生了改变**：
>    - 你需要重新编译改变的源文件 `function1.cpp`，因为它的对象文件需要更新。
>    - 然后，用新生成的 `function1.o` 和旧的 `function2.o` 及 `main.o`（如果 `main.cpp` 和 `function2.cpp` 没变）链接生成最终的可执行文件。
>    - 如果只有 `function1.cpp` 发生变化，无需重新编译 `function2.cpp`，因为其对应的对象文件仍然是最新的。
>
> 简单来说，你总是需要重新编译所有改变了的源文件，以确保它们的对象文件是最新的。然后将所有相关的对象文件链接在一起，生成最终的可执行文件。未改变的源文件对应的对象文件可以重复使用，无需重新编译。

## 4. Meta Build- Cmake

==Cmake能够为我们编写makefile==

Cmake能够为我们提供跨平台的项目。

* 拥有Linux的C++程序与makefile并不能在windows上面构建该代码
* 拥有Windows上的visual studio解决方案也无法在Linux上面构建该代码

Cmake通过构建基于平台的系统文件解决了这个问题。



### 4.1 安装CMake

方法一

```bash
sudo apt install cmake
cmake --version
```

方法二：

cmake.org/download/

下载二进制发行版

![image-20240123213950354](./assets/image-20240123213950354.png)

进入下载文件夹

![image-20240123214017699](./assets/image-20240123214017699.png)

![image-20240123214047087](./assets/image-20240123214047087.png)

进入文件夹的bin，我们可以见到一个cmake 可执行文件

我们需要将这个可执行文件添加到系统中

即为，首先获取bin的路径

```bash
pwd
```

然后将这个路径复制到Home/.bashrc中

![image-20240123214318101](./assets/image-20240123214318101.png)







方法三：从官方网站下载source code

![image-20240123214417710](./assets/image-20240123214417710.png)

解压

![image-20240123214449029](./assets/image-20240123214449029.png)

进入这个文件夹，依次输入三个命令：

* ./bootstrap
* make
* sudo make install

![image-20240123214530099](./assets/image-20240123214530099.png)

如果boot出现错误，应该使用`./bootstrap -- -DCMAKE_USE_OPENSSL=OFF`，这是脱离OPENSSL的构建方式





## 4.2 Cmake构建流程

![image-20240124133434552](./assets/image-20240124133434552.png)





## 4.3 Cmake语法

### 4.3.1 C++源文件生成可执行程序的tool chain流程

![image-20240124133837943](./assets/image-20240124133837943.png)

不建议使用Cmake生成makefile，然后使用make命令。

建议

```bash
cmake -B 目录名(build)
cmake --build 目录名
```



### 4.3.2 Cmake流程图

![image-20240124134138466](./assets/image-20240124134138466.png)





### 4.3.3 Cmake命令行执行流程

![image-20240124134353437](./assets/image-20240124134353437.png)







### 4.3.4 Windows下的Cmake

* 首先安装Cmake

* 然后Cmake使用的是默认MSVC，你可以安装MinGW(gcc, clang)

* 使用cmake参数更改默认编译器

  ```c++
  cmake -G <generator-name> -T <toolset-spec> -A <platform-name><path-to-source>
  ```

  通过指定`MinGW Makefiles`来指定cmake使用gcc

下面是一个 CMake 命令的例子：

```bash
cmake -G "Visual Studio 16 2019" -A x64 ../source_dir
```

这个例子中的命令做了以下几件事情：

1. `-G "Visual Studio 16 2019"`：这告诉 CMake 使用 Visual Studio 2019 作为其生成器。CMake 将生成一个 Visual Studio 解决方案文件，可以用 Visual Studio 打开和构建。
2. `-A x64`：这指定了目标平台架构为 64 位（x64）。这告诉 CMake 生成一个针对 64 位系统的 Visual Studio 解决方案。
3. `../source_dir`：这是指向源代码目录的路径，CMake 会在这个目录下查找 `CMakeLists.txt` 文件。

请注意，`-T <toolset-spec>` 参数在这个例子中没有被使用，因为它是可选的，并且通常只在你需要指定一个特定版本的编译器或工具集时才会用到。例如，如果你想使用特定版本的 MSVC 编译器工具集，你可以添加 `-T` 选项指定它。

如果你确实需要指定工具集，比如你想使用不同于 Visual Studio 默认设置的编译器，你可以添加 `-T` 选项。例如，如果你想用 Visual Studio 2019 的 LLVM 工具集来编译项目，可以这样写：

```bash
cmake -G "Visual Studio 16 2019" -T LLVM -A x64 ../source_dir
```

这告诉 CMake 使用 LLVM 作为工具集来生成 Visual Studio 解决方案。





### 4.3.5 一个例子

```bash
cmake -B build
cmake --build build
build/Hello.exe
```

![image-20240124140646184](./assets/image-20240124140646184.png)





### 4.3.6 VScode中切换编译器

`cmake --help`能发现许多生成器，星号是默认的，我们一般倾向于使用MinGW Makefiles，因为微软的没开源



![image-20240124140926643](./assets/image-20240124140926643.png)



```bash
cmake -B build -G "MinGW Makefiles"
```

![image-20240124141144036](./assets/image-20240124141144036.png)











### 4.3.7 Cmake的组成与`.cmake`

* Cmake的执行从源树（CMakeLists.txt)的所在目录开始的

* Cmake命令行工具主要由五个可执行文件构成

  * cmake: `cmake -P filename.cmake`（这个命令针对.cmake文件，一般不会使用)

    * ```cmake
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
      ```

      ![image-20240124144616544](./assets/image-20240124144616544.png)

  * ctest

  * cpack

  * cmake-gui

  * ccmake











### 4.3.8 Cmake的变量操作`set`

* Cmake中的变量分为两种
  * Cmake本身提供的
  * 自定义的
* 变量名区分大小写
* 变量在存储时都是字符串
* 获取变量使用`${变量名}`
* 变量的基础操作是`set()`,`unset()`
  * 也可以用list或者string操作



#### `set()`

![image-20240124145025181](./assets/image-20240124145025181.png)



* `set(<variable><value>...[PARENT_SCOPE])`
* set可以给一个变量设置多个值
* 如果设置多个值，那么内部存储时使用`;`分割，但是显示时会直接连接着显示



```cmake

cmake_minimum_required(VERSION 3.20.0)


set(Var1 "YZZY")
message(${Var1})

# My Var因为带空格，所以作为变量名必须使用双引号，此时我们访问这个变量需要反编译符号\ 来表示空格
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

```

![image-20240124151132318](./assets/image-20240124151132318.png)











### 4.3.9 Cmake的变量操作`list`

```cmake
# 列表添加元素
list(APPEND <list> [<element>...])

# 列表删除元素
list(REMOVE_ITEM <list> <value> [value])

# 获取列表元素个数
list(LENGTH <list> <output variable>)

# 在列表中查找元素返回索引
list(FIND <list> <value> <out-var>)

# 在index位置插入
list(INSERT <list> <index> [<element> ...])

# 反转list
list(REVERSE <list>)

# 排序list
list(SORT <list> [...])

```





```cmake
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

```



![image-20240124154654986](./assets/image-20240124154654986.png)







### 4.3.10 流程控制

![image-20240124154738049](./assets/image-20240124154738049.png)

#### 基础语法

```cmake
if(<condition>)
<commands>
elseif(<condition>)
<command>
else()
<commands>
endif()
```

```cmake
foreach(<loop_var> RANGE <max>)
	command
endforeach()

foreach(<loop_var> RANGE <min> <max> [<step>])# step不设置默认步长为1
foreach(<loop_var> IN [LISTS <lists>][ITEMS<items>])# 逐元素遍历
```

```cmake
while(<condition>)
	<commands>
	endwhile()
```





#### 例子

```cmake
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

```

![image-20240124200625439](./assets/image-20240124200625439.png)

















### 4.3.11 函数

```cmake
function(<name>[<argument>...])
	<commands>
endfunction()
```



```cmake
cmake_minimum_required(VERSION 3.20.0)

# 这个函数的作用是设定仅有一个参数，然后依次做如下操作
# 输出函数名，输出参数值，修改参数值，再次输出参数值，使用固有变量输出三个参数值
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
MyFunc(${FirstArg} "value") # 我们可以发现，即使我们只设置了一个参数，我们仍然可以传入多个参数，被固有变量捕捉
# FirstArg没有改变代表函数的FirstArg只在函数内的作用域
message("FirstArg After Function =  ${FirstArg}")
```



![image-20240124202417707](./assets/image-20240124202417707.png)















### 4.3.12 作用域









## 4.5 Cmake构建项目的四种方式





## 4.6 静态库与动态库



## 4.7 Cmake与源文件交互



## 4.8 Cmake条件编译









