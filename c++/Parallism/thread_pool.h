#pragma once

# include <atomic>
# include <vector>
# include <iostream>
# include "thread_safe_queue.h"
# include <functional>

namespace my_thread_pool_basic {
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
		//用来停止所有线程的执行的标志，如果设置了，所有的线程都必须停止工作，代表我们将不再使用这个线程池
		std::atomic_bool done;

		//待处理工件队列，每个线程在完成当前任务后都会检查下一个线程。存入类型
		thread_safe_queue_space::thread_safe_queue<std::function<void()>> work_queue;
		
		//线程池
		std::vector<std::thread> threads;

		//保证对线程池中每一个线程的join
		join_threads joiner;

		void worker_thread() {
			//对于每个线程

			while (!done) {

				// 尝试从队列中取出任务
				// 若是成功取出任务，那么我们可以执行任务。
				// 若是取出任务失败，则可以yield这个线程，从而让给其他线程CPU空间。
				std::function<void()> task;
				if (work_queue.try_pop(task)) {

					task();
				}
				else {
					
					std::this_thread::yield();
				}
			}
		}
	public:

		thread_pool() :done(false), joiner(threads) {
			//我们拥有的线程数等于硬件支持的最大线程数
			const int thread_count = std::thread::hardware_concurrency();

			try {
				for (int i = 0; i < thread_count; i++) {
					// 第一个参数是一个指向成员函数worker_thread的指针，该成员函数定义了每个线程该做什么
					// 第二个参数指向的是这个线程池对象，这意味着新创建的线程将在这个线程池对象的上下文中运行worker_thread
					// 因为第一个参数调用的是成员函数，才需要第二个参数提供这个类的实例的上下文。
					// 调用的是如下构造函数
					/*
						template< class Function, class... Args >
						explicit thread( Function&& f, Args&&... args );
					*/

					threads.push_back(std::thread(&thread_pool::worker_thread, this));
				}
			}
			catch (...) {
				//包装线程的构造，如果发现异常可以简单的将done设置为true，这将指示其他线程结束运行。
				std::cout << "Error occurs, shut down all threads\n";
				done = true;
				throw;
			}
		}

		~thread_pool() {
			done = true;
		}

		template<typename Function_type>
		void submit(Function_type f) {
			//将任务推送到工作队列

			work_queue.push(std::function<void()>(f));
		}

	};

	void run() {
		//制造一百个任务分配给线程池
		thread_pool pool;
		std::cout << "Testing thread pool" << std::endl;

		for (int i = 0; i < 100; i++) {
			pool.submit([=] {
				printf("%d printed by thread - %d \n", i, std::this_thread::get_id());
				});
		}
		//在这个版本中，并没有等待所有任务完成的机制，因此将task全部压入后会直接调用析构函数，导致正在工作的线程全部停止。
		system("pause");

		//线程在等待一个任务完成（也就是说，在任务函数返回之前）时，不能去执行其他的任务。每个线程一次只能执行一个任务，直到这个任务完全完成后，线程才会检查是否有新的任务等待执行。在某些线程池的实现中，线程可以在执行一个长任务的同时，暂停该任务，去执行其他的（可能更短的）任务，然后再回来继续执行原任务。这种机制称为任务窃取（work stealing）。
	}
}


namespace my_thread_pool_with_waiting_threads {
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
		//用来停止所有线程的执行的标志，如果设置了，所有的线程都必须停止工作，代表我们将不再使用这个线程池
		std::atomic_bool done;

		//待处理工件队列，每个线程在完成当前任务后都会检查下一个线程。存入类型
		thread_safe_queue_space::thread_safe_queue<std::function<void()>> work_queue;//需要存入的对象copy constructable

		//线程池
		std::vector<std::thread> threads;

		//保证对线程池中每一个线程的join
		join_threads joiner;

		void worker_thread() {
			//对于每个线程

			while (!done) {

				// 尝试从队列中取出任务
				// 若是成功取出任务，那么我们可以执行任务。
				// 若是取出任务失败，则可以yield这个线程，从而让给其他线程CPU空间。
				std::function<void()> task;
				if (work_queue.try_pop(task)) {

					task();
				}
				else {

					std::this_thread::yield();
				}
			}
		}
	public:

		thread_pool() :done(false), joiner(threads) {
			//我们拥有的线程数等于硬件支持的最大线程数
			const int thread_count = std::thread::hardware_concurrency();

			try {
				for (int i = 0; i < thread_count; i++) {
					// 第一个参数是一个指向成员函数worker_thread的指针，该成员函数定义了每个线程该做什么
					// 第二个参数指向的是这个线程池对象，这意味着新创建的线程将在这个线程池对象的上下文中运行worker_thread
					// 因为第一个参数调用的是成员函数，才需要第二个参数提供这个类的实例的上下文。
					// 调用的是如下构造函数
					/*
						template< class Function, class... Args >
						explicit thread( Function&& f, Args&&... args );
					*/

					threads.push_back(std::thread(&thread_pool::worker_thread, this));
				}
			}
			catch (...) {
				//包装线程的构造，如果发现异常可以简单的将done设置为true，这将指示其他线程结束运行。
				std::cout << "Error occurs, shut down all threads\n";
				done = true;
				throw;
			}
		}

		~thread_pool() {
			done = true;
		}

		template<typename Function_type, typename... Args>
		auto sumbmit(Function_type f, Args&&... args) -> std::future<std::invoke_result_t<Function_type, Args...>>
		{
			typedef typename std::invoke_result_t<Function_type, Args... > result_type;
			std::packaged_task<result_type> task(std::move(f));
			std::future<result_type> res(task.get_future());

			work_queue.push(std::move(task));
			//这一句会报错，首先就是因为我们已经将callable function包装起来了，更深层因为task对应的packaged_task是movable而非copy constructable
			//基于更深层的原因，我们也不能使用thread_safe_queue_space::thread_safe_queue<std::packaged_task> work_queue;

			
			return res;
		}
	};
}