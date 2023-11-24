#pragma once
# include<thread>
# include<iostream>
# include<chrono>
namespace useful_functions {
	void foo() {
		printf("This is thread %d from foo using std function\n",std::this_thread::get_id());
	}

	void get_id_example() {
		// get_id()
		std::cout << "\n\n\nThis is from get_id() function" << std::endl;
		std::thread thread_1(foo);

		printf("This is thread %d outside thread using class function\n", thread_1.get_id());
		if(thread_1.joinable())
			thread_1.join();
		printf("This is unactive thread code outside thread using class function: code %d\n", thread_1.get_id());

		std::thread thread_2;
		printf("This is a thread using default constructor, its code is %d\n\n\n\n", thread_2.get_id());
		printf("--------------------------------------");
	}

	void sleep_for_example() {
		std::cout << "\n\n\nThis is from sleep_for() function" << std::endl;
		std::cout << "Sleep for 2 seconds..." << std::endl;

		// 暂停线程执行2秒
		std::this_thread::sleep_for(std::chrono::seconds(2));

		std::cout << "Woke up!" << std::endl;
		printf("\n\n\n--------------------------------------");
	}
	void threadFunction() {
		for (int i = 0; i < 10; ++i) {
			std::cout << "Thread " << std::this_thread::get_id() << " is yielding...\n";
			std::this_thread::yield(); // 提示调度器当前线程可以让出CPU
		}
	}
	void yeild_example() {
		std::cout << "\n\n\nThis is from yeild function" << std::endl;
		std::thread t1(threadFunction);
		std::thread t2(threadFunction);

		t1.join();
		t2.join();
		printf("\n\n\n--------------------------------------");
	}

	void hardware_concurrency_example() {
		std::cout << "\n\n\nThis is from hardware_concurrency function" << std::endl;
		int allowed_threads = std::thread::hardware_concurrency();
		printf("Allowed thread count in my device : %d\n", allowed_threads);
		printf("\n\n\n--------------------------------------");
	}
	void run() {
		get_id_example();
		sleep_for_example();
		yeild_example();
		hardware_concurrency_example();

	}
}