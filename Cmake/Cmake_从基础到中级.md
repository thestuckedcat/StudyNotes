让我们从一个非常基本的CMake配置开始，并逐步添加命令，以展示每个命令的作用和效果。假设我们有一个简单的项目结构，它包括一个主函数文件`main.cpp`，以及一些位于`include`和`src`目录下的头文件和源文件。

### 最基本的CMake配置

假设我们的项目只有一个文件：`main.cpp`。

**CMakeLists.txt**:

```cmake
cmake_minimum_required(VERSION 3.10)  # 指定CMake的最低版本要求
project(SimpleProject)                # 定义项目名称

add_executable(main main.cpp)         # 从main.cpp文件创建一个名为main的可执行文件
```

这是构建一个简单项目的最基础的CMake配置。它定义了项目名，以及一个基于`main.cpp`的可执行目标。

### 添加头文件目录

假设`main.cpp`包含了一些头文件，这些头文件位于项目的`include`目录下。

**修改后的CMakeLists.txt**:

```cmake
cmake_minimum_required(VERSION 3.10)
project(SimpleProject)

include_directories(include)         # 添加头文件搜索路径

add_executable(main main.cpp)
```

通过`include_directories(include)`，我们告诉CMake在编译时添加`include`目录到编译器的头文件搜索路径中。这样编译器就能找到`main.cpp`中通过`#include`引用的头文件。

### 包含多个源文件

假设我们的项目增长了，现在包含多个`.cpp`文件，这些文件位于`src`目录下。

**修改后的CMakeLists.txt**:

```cmake
cmake_minimum_required(VERSION 3.10)
project(SimpleProject)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")        # 自动收集所有在src目录下的.cpp文件

add_executable(main ${SOURCES})       # 使用收集到的源文件创建可执行文件
```

通过`file(GLOB SOURCES "src/*.cpp")`命令，CMake会自动找到`src`目录下所有的`.cpp`文件，并将这些文件路径存储在变量`SOURCES`中。然后，我们使用这个变量来指定`add_executable`命令应包含哪些源文件。

### 添加库文件

假设我们的项目需要链接到一个名为`mylib`的外部静态库，该库的头文件位于`/path/to/mylib/include`，库文件位于`/path/to/mylib/lib`。

**修改后的CMakeLists.txt**:

```cmake
cmake_minimum_required(VERSION 3.10)
project(SimpleProject)

include_directories(include /path/to/mylib/include)   # 添加外部库的头文件目录

link_directories(/path/to/mylib/lib)                  # 添加外部库的库文件目录

file(GLOB SOURCES "src/*.cpp")
add_executable(main ${SOURCES})

target_link_libraries(main mylib)                     # 链接外部库
```

在这里，我们使用`link_directories(/path/to/mylib/lib)`来告诉链接器在哪里找到`mylib`库文件，而`target_link_libraries(main mylib)`则指定了可执行文件`main`需要链接到`mylib`库。

### 组织代码：添加子目录

如果项目进一步增长，我们可能希望将其组织成多个子模块，==每个子模块有自己的`CMakeLists.txt`文件==。

假设我们有一个名为`module`的子模块，位于项目的`modules/module`目录下。

**项目根目录下的CMakeLists.txt**:

```cmake
cmake_minimum_required(VERSION 3.10)
project(SimpleProject)

include_directories(include /path/to/mylib/include)
link_directories(/path/to/mylib/lib)

add_subdirectory(modules/module)     # 添加子模块

file(GLOB SOURCES "src/*.cpp")
add_executable(main ${SOURCES})
target_link_libraries(main mylib module)  # 链接到module子模块
```

**modules/module/CMakeLists.txt**:

```cmake
add_library(module module.cpp)       # 假设module模块只有一个module.cpp文件
```

通过在项目根目录的`CMakeLists.txt`文件中使用`add_subdirectory(modules/module)`，CMake会处理指定子目录中的`CMakeLists.txt`，这允许我们构建并链接到`module`库。

> ### `add_library(name [STATIC|SHARED|MODULE] sources...)`
>
> - **作用**：从指定的源文件创建一个库（静态库、动态库或模块库）。
> - **是否必须**：如果项目需要生成库文件以供其他项目或程序使用，或者组织代码时希望模块化，则此命令是必需的。

### 总结

这些步骤展示了如何逐步构建一个基础但完整的CMake项目配置。在所有这些命令中：

- `cmake_minimum_required`、`project`和`add_executable`（或`add_library`）是构建任何项目时的基础和必需部分。
- `include_directories`、`link_directories`、`target_link_libraries`根据是否有外部依赖和项目的组织结构，可能是必需的。
- `file(GLOB...)`和`add_subdirectory`提供了额外的便利和模块化能力，但并非严格必需，取决于项目的具体需求。





## 命令简介

### 1. `include_directories(dirs...)`

- **作用**：向编译器添加头文件搜索路径。这告诉编译器在编译源代码文件时，在这些目录中查找`#include`指令指定的头文件。
- **是否必须**：如果项目中的源文件需要引用位于非标准位置的头文件，则此命令是必需的。

### 2. `link_directories(dirs...)`

- **作用**：向链接器添加库文件搜索路径。这指示链接器在链接生成可执行文件或库文件时，在这些目录中查找外部库文件。
- **是否必须**：如果项目依赖的库不在链接器的默认搜索路径中，则此命令是必需的。

### 3. `file(GLOB|GLOB_RECURSE variable [RELATIVE path] patterns...)`

- **作用**：将匹配特定模式的文件路径赋值给变量。常用于自动收集项目源文件列表。
- **是否必须**：不是必需的。手动指定源文件或使用其他命令也可以达到同样的目的，但`file(GLOB...)`可以简化文件管理。

### 4. `add_library(name [STATIC|SHARED|MODULE] sources...)`

- **作用**：从指定的源文件创建一个库（静态库、动态库或模块库）。
- **是否必须**：如果项目需要生成库文件以供其他项目或程序使用，或者组织代码时希望模块化，则此命令是必需的。

### 5. `add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])`

- **作用**：向项目添加一个子目录，==这个目录下的`CMakeLists.txt`文件也会被处理。==
- **是否必须**：对于模块化项目结构，尤其是包含多个组件或子模块的大型项目，使用此命令可以提高项目组织的清晰度和管理的便利性。

### 6. `add_executable(name sources...)`

- **作用**：从指定的源文件创建一个可执行文件。
- **是否必须**：对于生成可执行程序的项目，此命令是必需的。

### 7. `target_link_libraries(target [item1 [item2 [...]]] [[debug|optimized|general] item]...)`

- **作用**：为目标（可执行文件或库）指定链接库。这告诉链接器在链接时包含这些库。
- **是否必须**：如果项目中的代码依赖于外部库（如第三方库或系统库），则此命令是必需的。









## 直接在`add_library`或`add_executable`中添加所有源文件

这种方法适用于以下情况：

- **项目规模较小**：当项目相对较小，源文件数量可管理时，直接列出所有文件可能更简单直接。
- **构建配置简单**：如果项目的构建配置比较直接，没有太多复杂的条件编译或特殊的构建需求，直接添加源文件可能更方便。

### 使用`add_subdirectory`添加子目录

这种方法适用于以下情况：

- **模块化项目结构**：对于大型项目，尤其是那些逻辑上分为多个子模块或组件的项目，使用`add_subdirectory`可以帮助组织代码，每个子模块或组件都有自己的`CMakeLists.txt`，从而实现更清晰的项目结构。
- **重用和独立测试**：如果项目中的某些部分（如库）需要被独立构建或测试，或可能被其他项目重用，将它们放在自己的目录中，并使用`add_subdirectory`引入，会更加合适。
- **复杂构建逻辑**：对于需要复杂构建逻辑的项目，比如不同平台下有不同的构建选项，或者包含了条件编译的部分，使用子目录结构可以让构建逻辑更加模块化，每部分的构建配置可以独立管理，增加了构建系统的灵活性和可维护性。