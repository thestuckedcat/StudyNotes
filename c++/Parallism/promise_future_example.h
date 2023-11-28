#pragma once
# include <iostream>
# include <functional>
# include <thread>
# include <future>
# include <stdexcept>
# include <chrono>

namespace promise_future_example {
	void print_int(std::future<int>& fut) {
		std::cout << "waiting for value from print thread\n";
		std::cout << "value: " << fut.get() << '\n';
	}

	void run() {
		/*这个代码让future在独立thread中，然后在main thread中set_value来传递值结束这个线程*/
		std::promise<int> prom;
		std::future<int> fut = prom.get_future();//绑定

		std::thread print_thread(print_int, std::ref(fut));

		std::this_thread::sleep_for(std::chrono::milliseconds(5000));

		std::cout << "setting the value in main thread \n";
		prom.set_value(10);

		print_thread.join();


	}

	//使用package的版本,在这个版本中实现了不同线程的传递数据，以及future和promise和packaged_task的协作
	int add(int x, std::future<int>& fut) {
		std::cout << "add function run in : " << std::this_thread::get_id() << std::endl;

		std::cout << "waiting values\n";
		int ans = fut.get() + x;
		std::cout << "values get\n";
		return ans;
	}

	void task_thread() {
		std::packaged_task<int(int, std::future<int>&)> task_1(add);
		std::future<int> future_for_task = task_1.get_future();
		std::promise<int> prom;
		std::future<int> future_for_promise = prom.get_future();

		std::thread thread_1(std::move(task_1), 5, std::ref(future_for_promise));
		thread_1.detach();
		prom.set_value(10);
		
		std::cout << "task thread - " << future_for_task.get() << "\n";
	}

	void run_with_package() {
		task_thread();
		std::this_thread::sleep_for(std::chrono::seconds(10));//防止main过短提前结束导致thread_1还没完成
	}
}