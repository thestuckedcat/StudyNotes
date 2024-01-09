#pragma once
# include<iostream>
# include<mutex>
# include<queue>
# include <memory>
# include <condition_variable>
# include <thread>

namespace thread_safe_queue_space {
	template<typename T>
	class thread_safe_queue {
		std::mutex m;
		std::condition_variable cv;
		std::queue<std::shared_ptr<T>> queue;//Ϊ�˱�����Ϊ��ԴԼ���������쳣�����ʹ����shared_ptr
	public:

		thread_safe_queue()
		{}

		thread_safe_queue(thread_safe_queue const& other_queue)
		{
			std::lock_guard<std::mutex> lg(other_queue.m);
			queue = other_queue.queue;
		}

		// ��ֵ���ð汾
		void push(const T& value) {
			std::lock_guard<std::mutex> lg(m);
			queue.push(std::make_shared<T>(value));
			cv.notify_one(); 
		}

		// ��ֵ���ð汾
		void push(T&& value) {
			std::cout << "��ֵ����" << std::endl;
			std::lock_guard<std::mutex> lg(m);
			queue.push(std::make_shared<T>(std::move(value)));
			cv.notify_one();
		}

		std::shared_ptr<T> pop() //combine pop and front
		{
			std::lock_guard<std::mutex> lg(m);
			if (queue.empty()) //����pop��empty�ľ���
			{
				return std::shared_ptr<T>();
			}
			else {
				std::shared_ptr<T> ref(queue.front());
				queue.pop();
				return ref;
			}
		}
		

		bool empty() {
			std::lock_guard<std::mutex> lg(m);
			return queue.empty();
		}

		std::shared_ptr<T> wait_pop() //��ͬ��pop,wait_popּ��ʵ��һ��ָ����У����ǶԵ�ǰ״̬��queue pop�������ܹ�һ����Ҫpopִ�ж���ָ��
		{
			std::unique_lock<std::mutex> lg(m);
			cv.wait(lg, [this] {return !queue.empty(); });
			std::shared_ptr<T> ref = queue.front(); //������ʼ����Ч����ֱ�ӳ�ʼ��ref(queue.front())һ��
			queue.pop();
			return ref;
			/*
			��һ���߳���ִ������ȴ�������������ȡlock�����Ҽ������Ƿ�Ϊ��
			�����Ϊ�գ�����ζ�����������Ԫ�أ���ô��������һ����䣬Ҳ����front��pop��ͬʱ���ֶ����Ŀ���Ȩ(holding the lock)
			�������Ϊ�գ���ô������ʧ�ܣ������������Ψһ�������Ļ��������Է������������̵߳��ô˵ȴ�����Ȼ�����˯��

			��ˣ����ܳ�����ôһ�������������̶߳��ڵȴ���������һ��Ԫ��
			������push����Щ�̵߳����һ���ͻᱻ���ѣ���ô�Ψһ������������Ȼ���������ɹ���Ȼ�����ִ��
			��Ȼ������Ҳ����ʹ��notify_all�����������̶߳��������ѣ�����ֻ��һ���̻߳ᱻ�����������Ϊֻ��һһ���߳������ȡ���Ψһ����������
			�ڸ��̳߳�����Χ�󣬸��������ͷţ��Ա���һ�������ѵ��߳̿����ڻ�ȡ����notify_all��ȫ�����ѣ��ŶӼ��),��ʱ��һ���̻߳�ȡ���Ŀ���Ȩ����鷢�ֲ��У�����˯������Ҳ������notify_all������̫��
			*/

		}
		size_t size()
		{
			std::lock_guard<std::mutex> lg(m);
			return queue.size();
		}
		//�����Ƿ��ظò����Ƿ�ִ�гɹ��İ汾��ͨ����������ȡpop��ֵ
		bool wait_pop(T& ref)
		{
			std::unique_lock<std::mutex> lg(m);
			cv.wait(lg, [this] {
				return !queue.empty();
				});

			ref = *(queue.front().get());
			queue.pop();
			return true;
		}

		bool pop(T& ref)
		{
			std::lock_guard<std::mutex> lg(m);
			if (queue.empty())
			{
				return false;
			}
			else
			{
				ref = queue.front();
				queue.pop();
				return true;
			}
		}

		// �汾1: ����std::shared_ptr<T>
		std::shared_ptr<T> try_pop() {
			std::lock_guard<std::mutex> lg(m);
			if (queue.empty()) {
				return std::shared_ptr<T>();
			}

			std::shared_ptr<T> ref(queue.front());
			queue.pop();
			return ref;
		}

		//�汾2: ͨ�����÷���ֵ�������ز����Ƿ�ɹ���boolֵ
		bool try_pop(T& value) {
			std::lock_guard<std::mutex> lg(m);

			if (queue.empty()) {
				return false;
			}

			value = *(queue.front());
			queue.pop();
			return true;
		}

	};


	//����

	void producer(thread_safe_queue<int>& queue) {
		/*����һ��ѭ��������һϵ�е��������� 0 �� 9����������Щ�������͵������ thread_safe_queue �����С�ÿ�����ͺ�������ͣһС��ʱ�䣨100���룩*/
		for (int i = 0; i < 10; ++i) {
			std::cout << "Producing " << i << std::endl;
			queue.push(i);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}

	void consumer(thread_safe_queue<int>& queue) {
		/*�����ԴӶ�����ȡ��Ԫ�أ�����ÿ�γ��Ժ���ͣһС��ʱ�䣨150���룩���������ʹ���� wait_pop ����������ζ���������Ϊ�գ��������߳̽��ȴ�ֱ����������Ԫ�ؿɹ����ѡ�һ����������Ԫ�أ��������߳̽�ȡ��Ԫ�ز���ӡ����*/
		for (int i = 0; i < 10; ++i) {
			auto item = queue.wait_pop();
			if (item) {
				std::cout << "Consumed " << *item << std::endl;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(150));
		}
	}
	void run() {
		thread_safe_queue<int> tsq;

		std::thread producer_thread(producer, std::ref(tsq));
		std::thread consumer_thread(consumer, std::ref(tsq));

		producer_thread.join();
		consumer_thread.join();
	}
}
