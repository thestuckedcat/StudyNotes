#pragma once
#include <chrono>
#include <iostream>
#include <thread>

namespace transfer_ownership {
	void foo() {
		//std::cout << "Thread ID " << std::this_thread::get_id() << " from foo\n";
		printf("Thread ID %d from foo\n", std::this_thread::get_id());
	}

	void bar() {
		//std::cout << "Thread ID " << std::this_thread::get_id() << " from bar\n";
		printf("Thread ID %d from bar\n", std::this_thread::get_id());
	}

	void run() {
		std::thread thread_1(foo);

		std::thread thread_2 = std::move(thread_1);

		thread_1 = std::thread(bar);
		//在这里，发生了隐式的移动调用（因为右边是右值）而非赋值，所以没问题

		/*
		std::thread_3(foo);
		thread_1 = std::move(thread_3);//这一个操作会throw 一个exception，因为thread1有管理的线程，这里的操作实际上是在覆盖所有权而非转移
		*/
		thread_1.join();
		thread_2.join();
	}
}