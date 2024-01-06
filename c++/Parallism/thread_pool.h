#pragma once

# include <atomic>
# include <vector>
# include <iostream>
# include "thread_safe_queue.h"
# include <functional>

namespace my_thread_pool {
	class join_threads {
		std::vector<std::thread>& threads;

	public:
		explicit join_threads(std::vector<std::thread>& _threads) :
			threads(_threads)
		{}

		~join_threads()
		{
			for (long i = 0; i < threads.size(); i++)
			{
				if (threads[i].joinable())
					threads[i].join();
			}
		}

	};

	class thread_pool {
		std::atomic_bool done;//用来停止所有线程的执行的标志，如果设置了，所有的线程都必须停止工作，代表我们将不再使用这个线程池

		thread_safe_queue_space::thread_safe_queue<std::function<void()>> work_queue;//待处理工件队列，每个线程在完成当前任务后都会检查下一个线程。存入类型
		
		join_threads joiner;//保证对线程池中每一个线程的join

		void worker_thread() {
			while (done) {
				std::function<void()> task;
				if (work_queue.try_pop(task)) {
					task();
				}
				else {
					std::this_thread::yield;
				}
			}
		}
	public:

	};

	void run() {

	}
}