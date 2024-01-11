#pragma once

# include <atomic>
# include <vector>
# include <iostream>
# include "thread_safe_queue.h"
# include <functional>
# include <algorithm>
# include <list>

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

	
	// function_wrapper 是一个类型擦除的工具类，用于封装和调用任何可调用对象（如函数、Lambda 表达式和函数对象）。
	class function_wrapper {
		// impl_base 是一个内部抽象基类，用于定义统一的调用接口。
		struct impl_base {
			virtual void call() = 0;
			virtual ~impl_base() {}
		};

		// impl_type 是一个模板子类，继承自 impl_base。
		// 它封装了一个具体的可调用对象。
		template<typename F>
		struct impl_type : impl_base {
			F f; 
			impl_type(F&& f_) : f(std::move(f_)) {}
			void call() { f(); }
		};

		// impl 是一个指向 impl_base 的 unique_ptr，用于存储具体的 impl_type 实例。
		std::unique_ptr<impl_base> impl;

	public:
		// 构造函数，接受任何可调用对象，并将其封装在 impl_type 中。
		template<typename F>
		function_wrapper(F&& f) :
			impl(new impl_type<F>(std::move(f)))
		{}

		// 调用运算符，通过 impl 指针调用实际的可调用对象。
		void operator()() { impl->call(); }

		// 默认构造函数。
		function_wrapper() {}

		// 移动构造函数，允许将 function_wrapper 对象之间进行移动。
		function_wrapper(function_wrapper&& other) :
			impl(std::move(other.impl))
		{}

		// 移动赋值运算符，允许将 function_wrapper 对象之间进行移动。
		function_wrapper& operator=(function_wrapper&& other) {
			impl = std::move(other.impl);
			return *this;
		}

		// 删除拷贝构造函数和拷贝赋值运算符，防止对象被拷贝。
		function_wrapper(const function_wrapper&) = delete;
		function_wrapper(function_wrapper&) = delete;
	};
	



	class thread_pool {
		//用来停止所有线程的执行的标志，如果设置了，所有的线程都必须停止工作，代表我们将不再使用这个线程池
		std::atomic_bool done;

		//待处理工件队列，每个线程在完成当前任务后都会检查下一个线程。存入类型
		//thread_safe_queue_space::thread_safe_queue<std::function<void()>> work_queue;//需要存入的对象copy constructable
		//我们需要自己写一个函数包装器
		thread_safe_queue_space::thread_safe_queue<function_wrapper> work_queue;

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
				//std::function<void()> task;
				function_wrapper task;
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

		/*template<typename Function_type, typename... Args>
		auto submit(Function_type f, Args&&... args) -> std::future<std::invoke_result_t<Function_type, Args...>> {
			typedef std::invoke_result_t<Function_type, Args...> result_type;

			// 创建 std::packaged_task，绑定函数和参数
			std::packaged_task<result_type()> task(
				std::bind(std::forward<Function_type>(f), std::forward<Args>(args)...)
			);
			std::future<result_type> res = task.get_future();

			// 将任务移动到工作队列
			work_queue.push(std::move(task));

			return res;
		}*/
		template<typename Function_type>
		auto submit(Function_type f) -> std::future<std::invoke_result_t<Function_type>>
		{
			typedef typename std::invoke_result_t<Function_type> result_type;
			std::packaged_task<result_type()> task(f);
			std::future<result_type> res=task.get_future();

			work_queue.push(std::move(task));


			return res;
		}



		// 在waiting task案例中使用
		void run_pending_task() {
			function_wrapper task;
			if (work_queue.try_pop(task)) {
				task();
			}
			else {
				std::this_thread::yield();
			}
		}
	};

	// 每个线程自己进行的加法
	template<typename Iterator, typename T>
	struct accumulate_block
	{
		T operator()(Iterator first, Iterator last)
		{
			T value = std::accumulate(first, last, T());//以T的默认初始构造作为初始值
			printf(" %d - %d  \n", std::this_thread::get_id(), value);
			return value;
		}
	};

	//
	template<typename Iterator, typename T>
	T parallel_accumulate(Iterator first, Iterator last, T init)
	{
		unsigned long const length = std::distance(first, last);
		thread_pool pool;

		if (!length)
			return init;

		// 计算任务数量
		unsigned long const min_per_thread = 25;
		unsigned long const max_threads =
			(length + min_per_thread - 1) / min_per_thread;

		unsigned long const hardware_threads =
			std::thread::hardware_concurrency();

		unsigned long const num_threads =
			std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);

		unsigned long const block_size = length / num_threads;


		//分配任务给thread pool
		std::vector<std::future<T> > futures(num_threads - 1);

		Iterator block_start = first;
		for (unsigned long i = 0; i < (num_threads - 1); ++i)
		{
			Iterator block_end = block_start;
			std::advance(block_end, block_size);
			//提交
			futures[i] = pool.submit(std::bind(accumulate_block<Iterator, T>(), block_start, block_end));//move
			block_start = block_end;
		}
		T last_result = accumulate_block<int*, int>()(block_start, last);


		//等待任务完成
		T result = init;
		for (unsigned long i = 0; i < (num_threads - 1); ++i)
		{
			result += futures[i].get();
		}
		result += last_result;
		return result;
	}

	void run() {
		const int size = 1000;
		int* my_array = new int[size];
		srand(0);


		for (size_t i = 0; i < size; i++) {
			my_array[i] = 1;
		}

		long result = parallel_accumulate<int*, int>(my_array, my_array + size, 0);
		std::cout << "final sum is -" << result << std::endl;
	}



	// thread pool waiting other tasks


	// 并行版本的quick sort
	template<typename T>
	struct sorter
	{
		thread_pool pool;

		std::list<T> do_sort(std::list<T> & chunk_data)
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

	void run_quick_sort() {
		const int size = 900;
		std::list<int> my_array;
		srand(0);

		for (size_t i = 0; i < size; i++) {
			my_array.push_back(rand());
		}

		my_array = parallel_quick_sort(my_array);
		for (size_t i = 0; i < size; i++) {
			std::cout << my_array.front() << std::endl;
			my_array.pop_front();
		}
	}
}