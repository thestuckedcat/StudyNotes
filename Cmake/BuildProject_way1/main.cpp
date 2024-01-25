# include<iostream>
# include "animal/dog.h"
int main(int argc, char const *argv[]){

  Dog dog;
  std::cout << dog.barking() << std::endl;
  return 0;
}