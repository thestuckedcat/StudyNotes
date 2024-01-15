#pragma once

# include <atomic>
# include <vector>
# include <iostream>
# include <future>
# include <numeric>
# include <list>
# include <deque>
# include <mutex>
# include "thread_safe_queue.h"

namespace thread_pool_with_work_stealing {

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

	class function_wrapper {
		struct impl_base {
			virtual void call() = 0;
			virtual ~impl_base() {}
		};

		template<typename F>
		struct impl_type : impl_base {
			F f;
			impl_type(F&& f_) : f(std::move(f_)) {}
			void call() { f(); }
		};

		std::unique_ptr<impl_base> impl;

	public:
		template<typename F>
		function_wrapper(F&& f) :
			impl(new impl_type<F>(std::move(f)))
		{}
		void operator()() { impl->call(); }

		function_wrapper() {}

		function_wrapper(function_wrapper&& other) :
			impl(std::move(other.impl))
		{}

		function_wrapper& operator=(function_wrapper&& other) {
			impl = std::move(other.impl);
			return *this;
		}

		function_wrapper(const function_wrapper&) = delete;
		function_wrapper(function_wrapper&) = delete;
	};

	class work_stealing_queue {

		std::deque<function_wrapper> the_queue;
		mutable std::mutex the_mutex;//empty是const函数，但是它会改变the_mutex的状态，因此需要使用mutable保证the_mutex能被改变

	public:
		work_stealing_queue() {}

		//因为work_stealing_queue的 mutex既不是copy constructable也不是copy assignable，因此直接杜绝这个class
		work_stealing_queue(const work_stealing_queue& other) = delete;
		work_stealing_queue& operator=(const work_stealing_queue& other) = delete;

		void push(function_wrapper data) {
			std::lock_guard<std::mutex> lock(the_mutex);
			the_queue.push_front(std::move(data));

		}

		bool empty() const {
			std::lock_guard<std::mutex> lock(the_mutex);
			return the_queue.empty();
		}

		bool try_pop(function_wrapper& res) {
			std::lock_guard<std::mutex> lock(the_mutex);
			if (the_queue.empty()) {
				return false;
			}
			res = std::move(the_queue.front());
			the_queue.pop_front();
			return true;
		}

		bool try_steal(function_wrapper& res) {
			std::lock_guard<std::mutex> lock(the_mutex);
			if (the_queue.empty()) {
				return false;
			}

			res = std::move(the_queue.back());
			the_queue.pop_back();
			return true;

		}
	};

	class thread_pool_with_work_steal {

		std::atomic_bool done;
		// 线程池全局task队列
		thread_safe_queue_space::thread_safe_queue<function_wrapper> global_work_queue;

		// 本地队列变为新的queue，同时将访问各个thread本地queue的指针存在父级结构
		// static thread_local std::unique<std::queue<function_wrapper>> local_work_queue;
		std::vector<std::unique_ptr<work_stealing_queue>> queues;

		// 线程池线程
		std::vector<std::thread> threads;
		join_threads joiner;

		//本地队列及其标识
		static thread_local work_stealing_queue* local_work_queue;
		static thread_local unsigned my_index;

		void worker_thread(unsigned my_index_) {
			my_index = my_index_;
			local_work_queue = queues[my_index].get();
			while (!done) {
				run_pending_task();
			}
		}

		bool pop_task_from_local_queue(function_wrapper& task) {
			return local_work_queue && local_work_queue->try_pop(task);
		}

		bool pop_task_from_pool_queue(function_wrapper& task) {
			return global_work_queue.try_pop(task);
		}

		bool pop_task_from_other_thread_queue(function_wrapper& task) {
			// 遍历队列向量中的所有队列，找到具有额外工作的local queue
			for (unsigned i = 0; i < queues.size(); i++) {
				//这里是因为，如果我们总是从第一个线程开始访问，我们就会增加第一个线程的负载，因此我们选择从下一个线程开始
				//如果发现有一个线程有任务，那就可以偷到自己的线程。
				unsigned const index = (my_index + i + 1) % queues.size();
				if (queues[index]->try_steal(task)) {
					return true;
				}
			}
			return false;
		}

	public:
		thread_pool_with_work_steal() :joiner(threads), done(false) {
			unsigned const thread_count = std::thread::hardware_concurrency();

			try {
				for (unsigned i = 0; i < thread_count; ++i) {
					queues.push_back(std::unique_ptr<work_stealing_queue>(new work_stealing_queue));
					threads.push_back(std::thread(&thread_pool_with_work_steal::worker_thread, this, i));
				}
			}
			catch (...)
			{
				std::cout << "Error occurs" << std::endl;
				done = true;
				throw;
			}

			
		}

		~thread_pool_with_work_steal()
		{
			done = true;
		}

		template<typename Function_type>
		auto submit(Function_type f) -> std::future<std::invoke_result_t<Function_type>>
		{
			// 当我们第一次向线程池submit时，caller thread并不是线程池线程，因此没有local_queue
			// submit在caller thread上向线程池push task
			// 如果这个task还会向同一个线程池push对象，那么此时caller thread就是线程池线程，此时提交到本地queue
			typedef typename std::invoke_result_t<Function_type> result_type;
			std::packaged_task<result_type()> task(f);
			std::future<result_type> res = task.get_future();

			//work_queue.push(std::move(task));
			//如果这个线程是池线程，那么我们可以简单地将任务推送到本地队列，避免争用全局工作队列
			//如果这个线程不是池线程，那么我们将任务推送到池工作队列中
			//即为初始call推送到池线程，递归调用使用本地queue
			//std::cout  << (*local_work_queue).size() << std::endl;
			if (local_work_queue) {
				(*local_work_queue).push(std::move(task));
			}
			else {
				global_work_queue.push(std::move(task));
			}


			return res;
		}

		void run_pending_task() {
			function_wrapper task;
			//首先尝试从本地队列获取任务
			//然后尝试从全局队列弹出任务
			//最后尝试窃取任务
			if (pop_task_from_local_queue(task) ||
				pop_task_from_pool_queue(task) ||
				pop_task_from_other_thread_queue(task)) {
				task();
			}
			else {
				std::this_thread::yield();
			}
		}
	};




	template<typename T>
	struct sorter {

		thread_pool_with_work_steal pool;

		std::list<T> do_sort(std::list<T>& chunk_data)
		{
			if (chunk_data.size() < 2)
				return chunk_data;

			std::list<T> result;
			result.splice(result.begin(), chunk_data, chunk_data.begin());
			T const& partition_val = *result.begin();

			typename std::list<T>::iterator divide_point = std::partition(chunk_data.begin(),
				chunk_data.end(), [&](T const& val)
				{
					return val < partition_val;
				});

			std::list<T> new_lower_chunk;
			new_lower_chunk.splice(new_lower_chunk.end(), chunk_data,
				chunk_data.begin(), divide_point);

			std::future<std::list<T>> new_lower =
				pool.submit(std::bind(&sorter::do_sort, this, std::move(new_lower_chunk)));

			std::list<T> new_higher(do_sort(chunk_data));

			result.splice(result.end(), new_higher);

			//while(new_lower.wait_for(std::chrono::seconds(0))== std::future_status::timeout)
			while (!new_lower._Is_ready())
			{
				pool.run_pending_task();
			}

			result.splice(result.begin(), new_lower.get());

			return result;

		}

	};

	template<typename T>
	std::list<T> parallel_quick_sort(std::list<T> input)
	{
		if (input.empty())
		{
			return input;
		}

		sorter<T> s;
		return s.do_sort(input);
	}

	void run_quick_sort();
}