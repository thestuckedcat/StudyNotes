#pragma once

# include <atomic>
# include <vector>
# include <iostream>
# include "thread_safe_queue.h"
# include <functional>
# include <algorithm>

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
		//����ֹͣ�����̵߳�ִ�еı�־����������ˣ����е��̶߳�����ֹͣ�������������ǽ�����ʹ������̳߳�
		std::atomic_bool done;

		//�����������У�ÿ���߳�����ɵ�ǰ����󶼻�����һ���̡߳���������
		thread_safe_queue_space::thread_safe_queue<std::function<void()>> work_queue;
		
		//�̳߳�
		std::vector<std::thread> threads;

		//��֤���̳߳���ÿһ���̵߳�join
		join_threads joiner;

		void worker_thread() {
			//����ÿ���߳�

			while (!done) {

				// ���ԴӶ�����ȡ������
				// ���ǳɹ�ȡ��������ô���ǿ���ִ������
				// ����ȡ������ʧ�ܣ������yield����̣߳��Ӷ��ø������߳�CPU�ռ䡣
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
			//����ӵ�е��߳�������Ӳ��֧�ֵ�����߳���
			const int thread_count = std::thread::hardware_concurrency();

			try {
				for (int i = 0; i < thread_count; i++) {
					// ��һ��������һ��ָ���Ա����worker_thread��ָ�룬�ó�Ա����������ÿ���̸߳���ʲô
					// �ڶ�������ָ���������̳߳ض�������ζ���´������߳̽�������̳߳ض����������������worker_thread
					// ��Ϊ��һ���������õ��ǳ�Ա����������Ҫ�ڶ��������ṩ������ʵ���������ġ�
					// ���õ������¹��캯��
					/*
						template< class Function, class... Args >
						explicit thread( Function&& f, Args&&... args );
					*/

					threads.push_back(std::thread(&thread_pool::worker_thread, this));
				}
			}
			catch (...) {
				//��װ�̵߳Ĺ��죬��������쳣���Լ򵥵Ľ�done����Ϊtrue���⽫ָʾ�����߳̽������С�
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
			//���������͵���������

			work_queue.push(std::function<void()>(f));
		}

	};

	void run() {
		//����һ�ٸ����������̳߳�
		thread_pool pool;
		std::cout << "Testing thread pool" << std::endl;

		for (int i = 0; i < 100; i++) {
			pool.submit([=] {
				printf("%d printed by thread - %d \n", i, std::this_thread::get_id());
				});
		}
		//������汾�У���û�еȴ�����������ɵĻ��ƣ���˽�taskȫ��ѹ����ֱ�ӵ��������������������ڹ������߳�ȫ��ֹͣ��
		system("pause");

		//�߳��ڵȴ�һ��������ɣ�Ҳ����˵��������������֮ǰ��ʱ������ȥִ������������ÿ���߳�һ��ֻ��ִ��һ������ֱ�����������ȫ��ɺ��̲߳Ż����Ƿ����µ�����ȴ�ִ�С���ĳЩ�̳߳ص�ʵ���У��߳̿�����ִ��һ���������ͬʱ����ͣ������ȥִ�������ģ����ܸ��̵ģ�����Ȼ���ٻ�������ִ��ԭ�������ֻ��Ƴ�Ϊ������ȡ��work stealing����
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

	class function_wrapper {
		struct impl_base {
			virtual void call() = 0;
			virtual ~impl_base() {}
		};

		template<typename F>
		struct impl_type : impl_base
		{
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

		function_wrapper()
		{}

		function_wrapper(function_wrapper&& other) :
			impl(std::move(other.impl))
		{}

		function_wrapper& operator=(function_wrapper&& other)
		{
			impl = std::move(other.impl);
			return *this;
		}

		function_wrapper(const function_wrapper&) = delete;
		function_wrapper(function_wrapper&) = delete;
	};

	class thread_pool {
		//����ֹͣ�����̵߳�ִ�еı�־����������ˣ����е��̶߳�����ֹͣ�������������ǽ�����ʹ������̳߳�
		std::atomic_bool done;

		//�����������У�ÿ���߳�����ɵ�ǰ����󶼻�����һ���̡߳���������
		//thread_safe_queue_space::thread_safe_queue<std::function<void()>> work_queue;//��Ҫ����Ķ���copy constructable
		//������Ҫ�Լ�дһ��������װ��
		thread_safe_queue_space::thread_safe_queue<function_wrapper> work_queue;

		//�̳߳�
		std::vector<std::thread> threads;

		//��֤���̳߳���ÿһ���̵߳�join
		join_threads joiner;

		void worker_thread() {
			//����ÿ���߳�

			while (!done) {

				// ���ԴӶ�����ȡ������
				// ���ǳɹ�ȡ��������ô���ǿ���ִ������
				// ����ȡ������ʧ�ܣ������yield����̣߳��Ӷ��ø������߳�CPU�ռ䡣
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
			//����ӵ�е��߳�������Ӳ��֧�ֵ�����߳���
			const int thread_count = std::thread::hardware_concurrency();

			try {
				for (int i = 0; i < thread_count; i++) {
					// ��һ��������һ��ָ���Ա����worker_thread��ָ�룬�ó�Ա����������ÿ���̸߳���ʲô
					// �ڶ�������ָ���������̳߳ض�������ζ���´������߳̽�������̳߳ض����������������worker_thread
					// ��Ϊ��һ���������õ��ǳ�Ա����������Ҫ�ڶ��������ṩ������ʵ���������ġ�
					// ���õ������¹��캯��
					/*
						template< class Function, class... Args >
						explicit thread( Function&& f, Args&&... args );
					*/

					threads.push_back(std::thread(&thread_pool::worker_thread, this));
				}
			}
			catch (...) {
				//��װ�̵߳Ĺ��죬��������쳣���Լ򵥵Ľ�done����Ϊtrue���⽫ָʾ�����߳̽������С�
				std::cout << "Error occurs, shut down all threads\n";
				done = true;
				throw;
			}
		}

		~thread_pool() {
			done = true;
		}

		/*template<typename Function_type, typename... Args>
		auto submit(Function_type&& f, Args&&... args) -> std::future<std::invoke_result_t<Function_type, Args...>>
		{
			typedef typename std::invoke_result_t<Function_type, Args... > result_type;
			std::packaged_task<result_type()> task(std::forward<Function_type>, std::forward<Args>(args)...);
			std::future<result_type> res(task.get_future());

			work_queue.push(std::move(task));

			
			return res;
		}*/
		template<typename Function_type>
		auto submit(Function_type f) -> std::future<std::invoke_result_t<Function_type>>
		{
			typedef typename std::invoke_result_t<Function_type> result_type;
			std::packaged_task<result_type()> task(f);
			std::future<result_type> res(task.get_future());

			work_queue.push(std::move(task));


			return res;
		}
	};

	// ÿ���߳��Լ����еļӷ�
	template<typename Iterator, typename T>
	struct accumulate_block
	{
		T operator()(Iterator first, Iterator last)
		{
			T value = std::accumulate(first, last, T());//��T��Ĭ�ϳ�ʼ������Ϊ��ʼֵ
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

		// ������������
		unsigned long const min_per_thread = 25;
		unsigned long const max_threads =
			(length + min_per_thread - 1) / min_per_thread;

		unsigned long const hardware_threads =
			std::thread::hardware_concurrency();

		unsigned long const num_threads =
			std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);

		unsigned long const block_size = length / num_threads;


		//���������thread pool
		std::vector<std::future<T> > futures(num_threads - 1);

		Iterator block_start = first;
		for (unsigned long i = 0; i < (num_threads - 1); ++i)
		{
			Iterator block_end = block_start;
			std::advance(block_end, block_size);
			//�ύ
			futures[i] = pool.submit(std::bind(accumulate_block<Iterator, T>(), block_start, block_end));//move
			block_start = block_end;
		}
		T last_result = accumulate_block<int*, int>()(block_start, last);


		//�ȴ��������
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
}