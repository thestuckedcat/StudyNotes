#pragma once

#include<vector>
#include<thread>
#include<iostream>
class thread_guard {
    std::thread& t;
public:
    explicit thread_guard(std::thread& _t) :t(_t) {
        std::cout << "Constructor is called, thread " << std::this_thread::get_id() << " is running" << std::endl;
    }

    ~thread_guard() {
        if (t.joinable()) {
            std::cout << "Destructor is called, thread "<< std::this_thread::get_id() << " is joined" << std::endl;
            t.join();
        }
    }

    thread_guard(thread_guard& const) = delete;
    thread_guard& operator= (thread_guard& const) = delete;
};
