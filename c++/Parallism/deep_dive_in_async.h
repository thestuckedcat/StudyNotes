#pragma once
# include <iostream>
# include <future>
# include <string>

namespace deep_dive_in_async {
	//以下三个method将在单独的异步中运行
	void printing() {
		std::cout << "printing runs on-" << std::this_thread::get_id() << std::endl;
	}

	int addition(int x, int y) {
		std::cout << "addition on-" << std::this_thread::get_id() << std::endl;
		return x + y;
	}

	int substract(int x, int y) {
		std::cout << "substract runs on-" << std::this_thread::get_id() << std::endl;
		return x - y;
	}

	//run
	void run() {
		std::cout << "main thread id -" << std::this_thread::get_id() << std::endl;

		int x = 100;
		int y = 50;

		std::future<void> f1 = std::async(std::launch::async, printing);
		std::future<int> f2 = std::async(std::launch::deferred, addition, x, y);
		std::future<int> f3 = std::async(std::launch::deferred | std::launch::async, substract, x, y);

		f1.get();
		std::cout << "value recieved using f2 future - " << f2.get() << std::endl;
		std::cout << "value recieved using f3 future - " << f3.get() << std::endl;

	}
}