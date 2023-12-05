#pragma once
#pragma once

#include <vector>
#include <thread>
#include <future>
#include <atomic>
#include <algorithm>
#include <execution>
namespace parallel_find_example {
	class join_threads {
		std::vector<std::thread>& threads;
	public:
		explicit join_threads(std::vector<std::thread>& _threads) :threads{ _threads } {}
		~join_threads() {
			for (long i = 0; i < threads.size(); i++) {
				if (threads[i].joinable()) {
					threads[i].join();
				}
			}
		}
	};


	template<typename Iterator, typename MatchType>
	Iterator parallel_find_pt(Iterator first, Iterator last, MatchType match)
	{
		struct find_element
		{
			void operator()(Iterator begin, Iterator end,
				MatchType match,
				std::promise<Iterator>* result,
				std::atomic<bool>* done_flag)
			{
				try
				{
					for (; (begin != end) && !std::atomic_load(done_flag); ++begin)
					{
						/*序列非空且done_flag为false(注意这里需要用atomic_load读取done_flag的值避免数据竞争*/
						if (*begin == match)
						{
							result->set_value(begin);
							//done_flag.store(true);
							std::atomic_store(done_flag, true);
							return;
						}
					}
				}
				catch (...)
				{
					result->set_exception(std::current_exception());
					done_flag->store(true); // 出现错误时也需要重置done_flag停止其他线程
				}
			}
		};

		unsigned long const length = std::distance(first, last);

		if (!length)
			return last;

		/*	Calculate the optimized number of threads to run the algorithm	*/

		unsigned long const min_per_thread = 25;
		unsigned long const max_threads = (length + min_per_thread - 1) / min_per_thread;

		unsigned long const hardware_threads = std::thread::hardware_concurrency();
		unsigned long const num_threads = std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);
		unsigned long const block_size = length / num_threads;

		/*	Declare the needed data structures	*/
		std::promise<Iterator> result;
		std::atomic<bool> done_flag(false);

		std::vector<std::thread> threads(num_threads - 1);

		{
			join_threads joiner(threads);

			// task dividing loop
			Iterator block_start = first;
			for (unsigned long i = 0; i < (num_threads - 1); i++)
			{
				Iterator block_end = block_start;
				std::advance(block_end, block_size);

				threads[i] = std::thread(find_element(), block_start, block_end, match, &result, &done_flag);

				block_start = block_end;
			}

			// perform the find operation for final block in this thread.
			find_element()(block_start, last, match, &result, &done_flag);
		}

		if (!done_flag.load())
		{
			return last;//未找到返回一个.end()，没有实际元素
		}

		return result.get_future().get();
		/*std::promise 对象即使没有绑定 std::future 也可以调用 set_value。当 set_value 被调用时，promise 对象会存储该值或异常，等待一个将来可能与之绑定的 future 对象来检索它。如果后续创建了一个 future 对象来与这个 promise 绑定，那么这个 future 就可以检索到之前 promise 中设置的值。

重要的是要注意，一旦 promise 设置了值或异常，再次对其调用 set_value 或 set_exception 将抛出异常。因此，确保只调用一次 set_value 或 set_exception 是很重要的。*/
	}






	template<typename Iterator, typename MatchType>
	Iterator parallel_find_async(Iterator first, Iterator last, MatchType match, std::atomic<bool>& done_flag)
	{
		try
		{
			unsigned long const length = std::distance(first, last);
			unsigned long const min_per_thread = 25;

			if (length < 2 * min_per_thread)
			{
				for (; (first != last) && done_flag; ++first)
				{
					if (*first == match)
					{
						done_flag = true;
						return first;
					}
				}
				return last;
			}
			else
			{
				//Iterator const mid_point = first + length / 2;
				Iterator const mid_point = std::next(first, length / 2);

				//后半部分并行处理
				std::future<Iterator> async_result =
					std::async(&parallel_find_async<Iterator, MatchType>, mid_point, last, match, std::ref(done_flag));
				//前半部分串行处理
				Iterator const direct_result =
					parallel_find_async(first, mid_point, match, std::ref(done_flag));

				return (direct_result == mid_point) ? async_result.get() : direct_result;//direct_result == mid_point时，代表first到mid_point-1找不到，此时目标只能从另一半找。
			}
		}
		catch (const std::exception&)
		{
			done_flag = true;
			throw;
		}
	}



	void print_results(const char* const tag,
		std::chrono::high_resolution_clock::time_point startTime,
		std::chrono::high_resolution_clock::time_point endTime) {

		printf("%s: Time : %fms\n", tag, std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(endTime - startTime).count());
		//std::chrono::duration<double, std::milli> 表示一个以毫秒为单位的时间间隔，并且这个时间间隔的值是一个双精度浮点数。
		//.count()：这是一个方法，用于获取 std::chrono::duration 对象内部存储的实际数值。在这种情况下，它返回转换为毫秒的时间间隔的双精度浮点数值。
	}


	const size_t testSize = 10000;

	void run()
	{

		std::vector<int> ints(testSize);
		for (size_t i = 0; i < testSize; i++)
		{
			ints[i] = i;
		}

		int looking_for = 50000000;

		auto startTime = std::chrono::high_resolution_clock::now();
		auto value = parallel_find_pt(ints.begin(), ints.end(), looking_for);
		auto endTime = std::chrono::high_resolution_clock::now();
		print_results("Parallel-package_task_impl :", startTime, endTime);

		startTime = std::chrono::high_resolution_clock::now();
		std::find(ints.begin(), ints.end(), looking_for);
		endTime = std::chrono::high_resolution_clock::now();
		print_results("STL sequntial :", startTime, endTime);

		startTime =std::chrono::high_resolution_clock::now();
		std::find(std::execution::par, ints.begin(), ints.end(), looking_for);
		endTime = std::chrono::high_resolution_clock::now();
		print_results("STL parallel-par :", startTime, endTime);

		startTime = std::chrono::high_resolution_clock::now();
		std::find(std::execution::seq, ints.begin(), ints.end(), looking_for);
		endTime = std::chrono::high_resolution_clock::now();
		print_results("STL parallel-seq :", startTime, endTime);


		startTime = std::chrono::high_resolution_clock::now();
		std::atomic<bool> done_flag{false};
		value = parallel_find_async(ints.begin(), ints.end(), looking_for, done_flag);
		endTime = std::chrono::high_resolution_clock::now();
		print_results("Parallel-async-impl :", startTime, endTime);

	}

}