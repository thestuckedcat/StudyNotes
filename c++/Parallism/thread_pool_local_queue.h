#pragma once

# include <atomic>
# include <vector>
# include <iostream>
# include "thread_safe_queue.h"
# include <functional>
# include <algorithm>
# include <list>
# include <memory>
# include <queue>
# include <future>
# include <numeric>

namespace my_thread_pool_with_local_queue {



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

	class thread_pool {
		std::atomic_bool done;

		thread_safe_queue_space::thread_safe_queue<function_wrapper> work_queue;

		// 每个线程自己的队列
		typedef std::queue<function_wrapper> local_queue_type;

		//每个线程都有自己的 local_work_queue 实例。这个实例在该线程第一次访问变量时创建，并在线程结束时销毁。
		static thread_local std::unique_ptr<local_queue_type> local_work_queue;

		std::vector<std::thread> threads;

		join_threads joiner;

		void worker_thread() {

			//这句代表创建时初始化，释放原来的对象并指向一个新对象
			local_work_queue.reset(new local_queue_type);

			while (!done) {
				run_pending_task();
			}
		}
	public:
		

		thread_pool() :done(false), joiner(threads) {
			const int thread_count = std::thread::hardware_concurrency();

			try {
				for (int i = 0; i < thread_count; i++) {
					//在被push到线程池的同时，这个线程就具有了local queue
					threads.push_back(std::thread(&thread_pool::worker_thread, this));
				}
			}
			catch (...) {
				std::cout << "Error occurs, shut down all threads\n";
				done = true;
				throw;
			}
		}

		~thread_pool() {
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
				work_queue.push(std::move(task));
			}


			return res;
		}



		// 在waiting task案例中使用
		void run_pending_task() {
			function_wrapper task;
			//如本地队列存在(不是nullptr)且不为空
			if (local_work_queue && !(*local_work_queue).empty()) {
				task = std::move((*local_work_queue).front());
				(*local_work_queue).pop();
				task();
			}
			else if (work_queue.try_pop(task)) {
				//本地队列无任务
				task();
			}
			else {
				//都没有任务
				std::this_thread::yield();
			}
		}
	};

	template<typename T>
	struct sorter
	{
		thread_pool pool;

		std::list<T> do_sort(std::list<T>& chunk_data)
		{
			if (chunk_data.size() < 2)
				return chunk_data;

			std::list<T> result;

			// void splice( const_iterator pos, list& other, const_iterator it );
			// Transfers the element pointed to by it from other into *this. The element is inserted before the element pointed to by pos.
			result.splice(result.begin(), chunk_data, chunk_data.begin());

			T const& partition_val = *result.begin();

			// ForwardIt partition( ForwardIt first, ForwardIt last, UnaryPredicate p );
			// Reorders the elements in the range [first, last) in such a way that all elements for which the predicate p returns true precede the elements for which predicate p returns false. Relative order of the elements is not preserved.
			typename std::list<T>::iterator divide_point = std::partition(chunk_data.begin(), chunk_data.end(), [&](T const& val) {

				return val < partition_val;
				});

			std::list<T> new_lower_chunk;
			new_lower_chunk.splice(new_lower_chunk.end(), chunk_data, chunk_data.begin(), divide_point);

			// 排序
			std::future<std::list<T>> new_lower = pool.submit(std::bind(&sorter::do_sort, this, std::move(new_lower_chunk)));

			std::list<T> new_higher(do_sort(chunk_data));

			//这种写法会造成死锁
			//result.splice(result.end(), new_higher);
			//result.splice(result.begin(), new_lower.get());

			result.splice(result.end(), new_higher);
			//手动暂停，为这个线程分配新的任务，直到某个线程把new_lower的活干了。
			while (!new_lower._Is_ready()) {
				pool.run_pending_task();
			}
			result.splice(result.begin(), new_lower.get());

			return result;
		}
	};

	template<typename T>
	std::list<T> parallel_quick_sort(std::list<T> input) {
		if (input.empty()) {
			return input;
		}
		sorter<T> s;
		
		return s.do_sort(input);
	}

	void run_quick_sort();

}