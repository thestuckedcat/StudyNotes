#pragma once
# include<iostream>
# include <thread>
# include <future>
# include <stdexcept>
# include <chrono>
# include <mutex>

namespace shared_future_example {
	

	// 因为对象失效而出错
	void print_result_with_error(std::future<int>& fut) {
		std::cout << fut.get() << "\n";
	}
	//一种看起来正确，实际上存在race condition的版本
	void print_result_with_race_condition(std::future<int>& fut) {
		if (fut.valid()) {
			std::cout << "this is valid future\n";
			std::cout << fut.get() << "\n";
		}
		else {
			std::cout << "this is invalid future\n";
		}

	}

	void run_1() {
		std::promise<int> prom;
		std::future<int> fut(prom.get_future());

		std::thread th1(print_result_with_race_condition, std::ref(fut));
		std::thread th2(print_result_with_race_condition, std::ref(fut));

		prom.set_value(5);

		th1.join();
		th2.join();
	}



	//使用shared_memory
	void print_result(std::shared_future<int>& fut) {
		std::cout << fut.get() << " - valid future\n";
	}

	void run_2() {
		std::promise<int> prom;
		std::shared_future<int> fut(prom.get_future());

		std::thread th1(print_result, std::ref(fut));
		std::thread th2(print_result, std::ref(fut));

		prom.set_value(5);

		th1.join();
		th2.join();
	}











}