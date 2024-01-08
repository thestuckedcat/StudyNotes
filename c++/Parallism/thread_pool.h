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


	class thread_pool {
		//����ֹͣ�����̵߳�ִ�еı�־����������ˣ����е��̶߳�����ֹͣ�������������ǽ�����ʹ������̳߳�
		std::atomic_bool done;

		//�����������У�ÿ���߳�����ɵ�ǰ����󶼻�����һ���̡߳���������
		thread_safe_queue_space::thread_safe_queue<std::function<void()>> work_queue;//��Ҫ����Ķ���copy constructable

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

		template<typename Function_type, typename... Args>
		auto sumbmit(Function_type f, Args&&... args) -> std::future<std::invoke_result_t<Function_type, Args...>>
		{
			typedef typename std::invoke_result_t<Function_type, Args... > result_type;
			std::packaged_task<result_type> task(std::move(f));
			std::future<result_type> res(task.get_future());

			work_queue.push(std::move(task));
			//��һ��ᱨ�����Ⱦ�����Ϊ�����Ѿ���callable function��װ�����ˣ��������Ϊtask��Ӧ��packaged_task��movable����copy constructable
			//���ڸ�����ԭ������Ҳ����ʹ��thread_safe_queue_space::thread_safe_queue<std::packaged_task> work_queue;

			
			return res;
		}
	};
}