#pragma once

# include <iostream>
# include <thread>
# include <atomic>

namespace thread_local_storage {
	std::atomic<int> i = 0;
	thread_local std::atomic<int> j {1};
	void foo() {
		++i;
		++j;
		printf("normal atomic i = %d, thread_local atomic j= %d\n",i.load(),j.load());
	}

	void run() {
		std::thread t1(foo);
		std::thread t2(foo);
		std::thread t3(foo);


		t1.join();
		t2.join();
		t3.join();

		std::cout << "i=" << i << "  j=" << j << std::endl;
	}
}