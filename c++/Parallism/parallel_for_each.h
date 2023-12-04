#pragma once

#include <vector>
#include <thread>
#include <future>
#include <iostream>
#include <chrono>
#include <execution>


namespace parallel_for_each {
	// 一些使用的工具
	void print_results(const char* const tag,
		std::chrono::high_resolution_clock::time_point startTime,
		std::chrono::high_resolution_clock::time_point endTime) {

		printf("%s: Time : %fms\n", tag, std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(endTime - startTime).count());
		//std::chrono::duration<double, std::milli> 表示一个以毫秒为单位的时间间隔，并且这个时间间隔的值是一个双精度浮点数。
		//.count()：这是一个方法，用于获取 std::chrono::duration 对象内部存储的实际数值。在这种情况下，它返回转换为毫秒的时间间隔的双精度浮点数值。
	}

	class join_threads {
		std::vector<std::thread>& threads;

	public:
		explicit join_threads(std::vector<std::thread>& _threads) :threads(_threads) {}

		~join_threads() {
			for (long i = 0; i < threads.size(); i++) {
				if (threads[i].joinable()) {
					threads[i].join();
				}
			}
		}
	};








	/* This is the parallel version of for_each function implmentation with package tasks and futures */
	template<typename Iterator, typename Func>
	void parallel_for_each_pt(Iterator first, Iterator last, Func f)
	{
		/*
		在package_task实现的版本中，我们分配thread资源，因此需要计算最优thread数量
		*/
		unsigned long const length = std::distance(first, last);

		if (!length)
			return;

		/*	Calculate the optimized number of threads to run the algorithm	*/

		unsigned long const min_per_thread = 25;//计算每个线程种最少需要处理的个数
		unsigned long const max_threads = (length + min_per_thread - 1) / min_per_thread;

		unsigned long const hardware_threads = std::thread::hardware_concurrency();
		unsigned long const num_threads = std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);
		unsigned long const block_size = length / num_threads;

		/*	Declare the needed data structures	*/

		std::vector<std::future<void>> futures(num_threads - 1);
		std::vector<std::thread> threads(num_threads - 1);
		join_threads joiner(threads);

		/*	Partition of data between threads	*/

		Iterator block_start = first;
		for (unsigned long i = 0; i <= (num_threads - 2); i++)
		{
			Iterator block_end = block_start;
			std::advance(block_end, block_size);

			std::packaged_task<void(void)> task(
				[=]()
				{
					std::for_each(block_start, block_end, f);
				}
			);

			futures[i] = task.get_future();
			threads[i] = std::thread(std::move(task));

			block_start = block_end;
		}

		// call the function for last block from this thread
		std::for_each(block_start, last, f);

		/*	wait until futures are ready	*/
		for (unsigned long i = 0; i < (num_threads - 1); ++i)
			futures[i].get();

	}

	/* This is the parallel version of for_each function implmentation with std::async */
	template<typename Iterator, typename Func>
	void parallel_for_each_async(Iterator first, Iterator last, Func f)
	{
		unsigned long const length = std::distance(first, last);

		if (!length)
			return;

		unsigned long const min_per_thread = 25;

		if (length < 2 * min_per_thread)
		{
			std::for_each(first, last, f);
		}
		else
		{
			Iterator const mid_point = first + length / 2;
			std::future<void> first_half =
				std::async(&parallel_for_each_async<Iterator, Func>, first, mid_point, f);

			parallel_for_each_async(mid_point, last, f);
			first_half.get();
		}

	}

	const size_t testSize = 1000;

	void run()
	{
		std::vector<int> ints(testSize);
		for (auto& i : ints) {
			i = 1;
		}

		auto long_function = [](const int& n)
		{
			int sum = 0;
			for (auto i = 0; i < 100000; i++)
			{
				sum += 1 * (i - 499);
			}
		};

		auto startTime = high_resolution_clock::now();
		std::for_each(ints.cbegin(), ints.cend(), long_function);
		auto endTime = high_resolution_clock::now();
		print_results("STL                   ", startTime, endTime);

		startTime = high_resolution_clock::now();
		for_each(std::execution::seq, ints.cbegin(), ints.cend(), long_function);
		endTime = high_resolution_clock::now();
		print_results("STL-seq               ", startTime, endTime);

		startTime = high_resolution_clock::now();
		std::for_each(std::execution::par, ints.cbegin(), ints.cend(), long_function);
		endTime = high_resolution_clock::now();
		print_results("STL-par               ", startTime, endTime);

		startTime = high_resolution_clock::now();
		parallel_for_each_pt(ints.cbegin(), ints.cend(), long_function);
		endTime = high_resolution_clock::now();
		print_results("Parallel-package_task ", startTime, endTime);

		startTime = high_resolution_clock::now();
		parallel_for_each_async(ints.cbegin(), ints.cend(), long_function);
		endTime = high_resolution_clock::now();
		print_results("Parallel-async        ", startTime, endTime);

		std::cout << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;

	}
}