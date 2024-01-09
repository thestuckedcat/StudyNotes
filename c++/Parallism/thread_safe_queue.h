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
		std::queue<std::shared_ptr<T>> queue;//为了避免因为资源约束产生的异常情况，使用了shared_ptr
	public:

		thread_safe_queue()
		{}

		thread_safe_queue(thread_safe_queue const& other_queue)
		{
			std::lock_guard<std::mutex> lg(other_queue.m);
			queue = other_queue.queue;
		}

		// 左值引用版本
		void push(const T& value) {
			std::lock_guard<std::mutex> lg(m);
			queue.push(std::make_shared<T>(value));
			cv.notify_one(); 
		}

		// 右值引用版本
		void push(T&& value) {
			std::cout << "右值引用" << std::endl;
			std::lock_guard<std::mutex> lg(m);
			queue.push(std::make_shared<T>(std::move(value)));
			cv.notify_one();
		}

		std::shared_ptr<T> pop() //combine pop and front
		{
			std::lock_guard<std::mutex> lg(m);
			if (queue.empty()) //避免pop和empty的竞争
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

		std::shared_ptr<T> wait_pop() //不同于pop,wait_pop旨在实现一个指令队列，不是对当前状态的queue pop，而是总共一定需要pop执行多少指令
		{
			std::unique_lock<std::mutex> lg(m);
			cv.wait(lg, [this] {return !queue.empty(); });
			std::shared_ptr<T> ref = queue.front(); //拷贝初始化，效果与直接初始化ref(queue.front())一样
			queue.pop();
			return ref;
			/*
			第一个线程来执行这个等待函数，它将获取lock，并且检查队列是否为空
			如果不为空，这意味着如果队列铀元素，那么将进行下一个语句，也就是front和pop，同时保持对锁的控制权(holding the lock)
			如果队列为空，那么此条件失败，它将解锁与此唯一锁关联的互斥锁（以方便允许其他线程调用此等待），然后进入睡眠

			因此，可能出现这么一种情况，即多个线程都在等待队列中有一个元素
			当我们push后，这些线程的随机一个就会被唤醒，获得此唯一锁关联的锁，然后检查条件成功，然后继续执行
			当然，我们也可以使用notify_all，这样所有线程都将被唤醒，但是只有一个线程会被允许继续，因为只有一一个线程允许获取与此唯一锁关联的锁
			在该线程超出范围后，该锁将被释放，以便另一个被唤醒的线程可以在获取锁（notify_all是全部唤醒，排队检查),此时另一个线程获取锁的控制权，检查发现不行，接着睡。这里也能体现notify_all的消耗太大
			*/

		}
		size_t size()
		{
			std::lock_guard<std::mutex> lg(m);
			return queue.size();
		}
		//以下是返回该操作是否执行成功的版本，通过引用来获取pop的值
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

		// 版本1: 返回std::shared_ptr<T>
		std::shared_ptr<T> try_pop() {
			std::lock_guard<std::mutex> lg(m);
			if (queue.empty()) {
				return std::shared_ptr<T>();
			}

			std::shared_ptr<T> ref(queue.front());
			queue.pop();
			return ref;
		}

		//版本2: 通过引用返回值，并返回操作是否成功的bool值
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


	//测试

	void producer(thread_safe_queue<int>& queue) {
		/*它在一个循环中生成一系列的整数（从 0 到 9），并将这些整数推送到传入的 thread_safe_queue 对象中。每次推送后，它会暂停一小段时间（100毫秒）*/
		for (int i = 0; i < 10; ++i) {
			std::cout << "Producing " << i << std::endl;
			queue.push(i);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}

	void consumer(thread_safe_queue<int>& queue) {
		/*它尝试从队列中取出元素，并在每次尝试后暂停一小段时间（150毫秒）。这个函数使用了 wait_pop 方法，这意味着如果队列为空，消费者线程将等待直到队列中有元素可供消费。一旦队列中有元素，消费者线程将取出元素并打印它。*/
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
