#pragma once
/*这一个部分主要展示线程参数传递*/
#include<iostream>
#include<thread>

namespace pass_parameters {
    void func1(int p, int q) {
        printf("X + Y = %d\n", p + q);
    }

    void func_2(int& x) {
        while (true) {
            printf("THread1 value of X -%d \n", x);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
    void run_1() {
        int p = 9;
        int q = 9;
        std::thread thread_1(func1, p, q);

        thread_1.join();
    }

    void run_2() {
        int x = 9;
        printf("Main thread value of X - %d\n", x);
        std::thread thread_2(func_2, std::ref(x));
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        x = 15;
        printf("Main thread value of X has been change to - %d \n", x);
        thread_2.join();
    }
}