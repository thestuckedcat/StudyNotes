# include<iostream>
# include "dog.h"
# include "cat.h"
# include "config.h"
int main(int argc, char const *argv[]){

  Dog dog;
  Cat cat;
  std::cout << dog.barking() << std::endl;
  std::cout << cat.barking() << std::endl;

  // 使用config的定义
  std::cout << CMAKE_CXX_STANDARD<<std::endl;
  return 0;
}