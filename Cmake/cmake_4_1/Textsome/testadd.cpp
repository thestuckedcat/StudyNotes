//You won't want to do this, u can complete headfile finding by cmakelist
//#include"../Libsome/adder.h"

#include"adder.h"
#include<iostream>
// 因为已经与include文件夹target include了，所以可以直接这样写
#include<GLFW/glfw3.h>
int main(){
    std::cout <<  MATH::add(2,3);



    GLFWwindow *window;

    if(!glfwInit()){
        fprintf(stderr, "Failed to initalize GLFW\n");
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(300, 300, "Gears", NULL, NULL);

    if(!window){
        fprintf(stderr, "Failed to open GLFW window\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    while(!glfwWindowShouldClose(window)){
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();


    return 0;
}



