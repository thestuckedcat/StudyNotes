#pragma once
#include<iostream>
#include<thread> // basic header file

namespace launch_a_thread {
    void func1() {
        std::cout << "Hello fron function1 - thread  " << std::this_thread::get_id() << std::endl;
    }

    class my_class {
    public:
        void operator()() {
            std::cout << "Hello from the class with overload call operator in thread " << std::this_thread::get_id() << std::endl;
        }
    };


    void run() {
        std::thread thread1(func1);
        my_class mc;
        std::thread thread2(mc);
        std::thread thread3([] {
            std::cout << "Hello from the lambda, in thread  " << std::this_thread::get_id() << std::endl;
            });

        thread1.join();
        thread2.join();
        thread3.join();
    }
}