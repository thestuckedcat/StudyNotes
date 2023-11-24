#pragma once
#include<iostream>
#include<thread>
namespace join_joinable_detach {
	void test() {
		std::cout << "Call from test, thread ID " << std::this_thread::get_id() << std::endl;
	}

	void foo() {
		std::this_thread::sleep_for(std::chrono::milliseconds(5000));
		std::cout << "Hello from foo\n";
	}

	void bar() {
		std::this_thread::sleep_for(std::chrono::milliseconds(5000));
		std::cout << "Hello from bar\n";
	}

	void run_join_joinable()
	{
		std::thread thread1(test);

		if (thread1.joinable()) {
			std::cout << "Thread 1 is joinable before join \n";
		}
		else {
			printf("Thread 1 is not joinable before join \n");
		}
		thread1.join();

		if (thread1.joinable()) {
			std::cout << "Thread 1 is joinable after join \n";
		}
		else {
			printf("Thread 1 is not joinable after join \n");
		}

	}

	void run_join_detach() {
		std::thread foo_thread(foo);
		std::thread bar_thread(bar);

		bar_thread.detach();
		std::cout << "this is after bar thread detach\n";

		foo_thread.join();
		std::cout << " This is after foo thread join\n";
	}
}