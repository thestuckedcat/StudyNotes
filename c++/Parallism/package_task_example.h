#pragma once

# include<iostream>
# include<future>
# include<numeric>
# include<thread>
# include<functional>


namespace package_task_example {
	/*在这个例子中，我们使用package_task打包这个add function*/
	int add(int x, int y) {
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		std::cout << "add function runs in : " << std::this_thread::get_id() << std::endl;
		return x + y;
	}

	void task_normal() {
		std::packaged_task<int(int, int)> task_1(add);
		std::future<int> future_1 = task_1.get_future(); // acquire the associated future with that package task 
		
		//不像async在构造后会自动运行，这里需要显式调用，否则future get会无法工作
		
		task_1(7, 8);									// 调用这个任务

		//在这个情况下，这个创建的任务将在当前线程中按顺序运行(同步执行），这种方式展现了package可以在任何线程中执行

		std::cout << "task normal - " << future_1.get() << "\n";
	}

	//如果想要异步运行任务，需要创建一个线程并将此任务传递给该线程
	void task_thread() {
		std::packaged_task<int(int, int)> task_1(add);
		std::future<int> future_1 = task_1.get_future();

		std::thread thread_1(std::move(task_1), 5, 6);//这样我们就可以异步的运行这个任务，package_task不是copyable的，因此必须要move
		thread_1.detach();							//异步的运行

		std::cout << "task thread - " << future_1.get() << "\n";
	}

	void run() {
		task_thread();
		task_normal();
		std::cout << "main thread id : " << std::this_thread::get_id() << std::endl;
	}
}