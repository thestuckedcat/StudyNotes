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

		// ÿ���߳��Լ��Ķ���
		typedef std::queue<function_wrapper> local_queue_type;

		//ÿ���̶߳����Լ��� local_work_queue ʵ�������ʵ���ڸ��̵߳�һ�η��ʱ���ʱ�����������߳̽���ʱ���١�
		static thread_local std::unique_ptr<local_queue_type> local_work_queue;

		std::vector<std::thread> threads;

		join_threads joiner;

		void worker_thread() {

			//��������ʱ��ʼ�����ͷ�ԭ���Ķ���ָ��һ���¶���
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
					//�ڱ�push���̳߳ص�ͬʱ������߳̾;�����local queue
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
			// �����ǵ�һ�����̳߳�submitʱ��caller thread�������̳߳��̣߳����û��local_queue
			// submit��caller thread�����̳߳�push task
			// ������task������ͬһ���̳߳�push������ô��ʱcaller thread�����̳߳��̣߳���ʱ�ύ������queue
			typedef typename std::invoke_result_t<Function_type> result_type;
			std::packaged_task<result_type()> task(f);
			std::future<result_type> res = task.get_future();

			//work_queue.push(std::move(task));
			//�������߳��ǳ��̣߳���ô���ǿ��Լ򵥵ؽ��������͵����ض��У���������ȫ�ֹ�������
			//�������̲߳��ǳ��̣߳���ô���ǽ��������͵��ع���������
			//��Ϊ��ʼcall���͵����̣߳��ݹ����ʹ�ñ���queue
			//std::cout  << (*local_work_queue).size() << std::endl;
			if (local_work_queue) {
				(*local_work_queue).push(std::move(task));
			}
			else {
				work_queue.push(std::move(task));
			}


			return res;
		}



		// ��waiting task������ʹ��
		void run_pending_task() {
			function_wrapper task;
			//�籾�ض��д���(����nullptr)�Ҳ�Ϊ��
			if (local_work_queue && !(*local_work_queue).empty()) {
				task = std::move((*local_work_queue).front());
				(*local_work_queue).pop();
				task();
			}
			else if (work_queue.try_pop(task)) {
				//���ض���������
				task();
			}
			else {
				//��û������
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

			// ����
			std::future<std::list<T>> new_lower = pool.submit(std::bind(&sorter::do_sort, this, std::move(new_lower_chunk)));

			std::list<T> new_higher(do_sort(chunk_data));

			//����д�����������
			//result.splice(result.end(), new_higher);
			//result.splice(result.begin(), new_lower.get());

			result.splice(result.end(), new_higher);
			//�ֶ���ͣ��Ϊ����̷߳����µ�����ֱ��ĳ���̰߳�new_lower�Ļ���ˡ�
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