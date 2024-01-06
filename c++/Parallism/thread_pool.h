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
		std::atomic_bool done;//����ֹͣ�����̵߳�ִ�еı�־����������ˣ����е��̶߳�����ֹͣ�������������ǽ�����ʹ������̳߳�

		thread_safe_queue_space::thread_safe_queue<std::function<void()>> work_queue;//�����������У�ÿ���߳�����ɵ�ǰ����󶼻�����һ���̡߳���������
		
		join_threads joiner;//��֤���̳߳���ÿһ���̵߳�join

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