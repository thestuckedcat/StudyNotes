首先，需要将visual studio的MSbuild加入Path

>  `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\msbuild.exe`

然后，在build目录下，直接使用

```bash
msbuild PROJECT_NAME.sln /p:Configuration=Debug
```

生成基于Debug编译的build，



在 Visual Studio 或 VSCode 中，当你使用 `msbuild` 命令或点击构建按钮时，构建系统会自动进行“增量构建”（incremental build）。这意味着它会比较源代码文件的修改时间和相应的编译输出（如 `.obj` 文件），只有当源文件较新，即被修改过后，才会重新编译这些文件。这个过程是自动完成的，不需要用户进行特殊的操作。



对于MSVC编译器，找到build/bin/Debug/<executename>.exe即可