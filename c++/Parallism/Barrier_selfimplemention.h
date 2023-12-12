#pragma once


# include <thread>
# include <atomic>
# include <condition_variable>
# include <barrier>
# include <iostream>
# include <vector>

namespace barrier_implemention {
	/*
	使用自旋等待方法实现barrier
	spin the threads until the required number of threads reached the barrier
	*/

	class barrier_spin {

		unsigned const thresh_hold; //需要等待的线程数
		std::atomic<unsigned> count;//当前等待的线程数
		std::atomic<unsigned> generation;//用于多次同步时，区分不同的同步点，如果只有一个同步点就不需要

	public:
		explicit barrier_spin(unsigned _thresh_hold) :thresh_hold(_thresh_hold), count(_thresh_hold), generation(0) {

		}

		void wait() {
			//等待线程数量到达要求
			//每个线程都会调用
			unsigned const my_generation = generation;

			if (!--count) {
				// 最后一个到达的线程重置计数器并增加 generation
				count = thresh_hold;
				++generation;
			}
			else {
				//其他的线程在忙等待
				while (generation == my_generation)
					std::this_thread::yield();
				/*
				让出执行权：调用 std::this_thread::yield() 会让当前执行的线程告诉操作系统它愿意放弃剩余的时间片，这给了操作系统机会去调度其他线程运行。但是，它仅仅是一个“建议”或“暗示”，操作系统可能会忽略这个请求。

				减少忙等待（busy-waiting）的影响：在某些算法中，线程可能会进入忙等待状态，即不断检查某个条件是否满足而不进行实际的有用工作。在这种情况下，使用 std::this_thread::yield() 可以减少对CPU资源的浪费，因为它允许操作系统在忙等待的线程和其他线程之间更有效地切换。

				提高多线程程序的响应性和公平性：在一些需要平衡多个线程执行的场景中，使用 std::this_thread::yield() 可以帮助提高程序的响应性和运行的公平性。
				
				在这个循环中，
				线程会不断检查 generation 是否等于 my_generation。如果条件成立，它就会调用 yield，这是一个向操作系统发出的信号，表示线程愿意让出其剩余的时间片。
				如果操作系统接受，那么这个线程会被暂时deactive，直到操作系统再次将其调度为活动状态。这个等待时间可能会有所不同，取决于操作系统的调度策略和系统上运行的其他任务。
				当操作系统最终再次调度该线程时，线程将从 yield 调用之后的代码继续执行。
				*/
			}
		}
	};


	/*
	使用共享变量
	*/
	class barrier_cond {

		std::mutex mMutex;
		std::condition_variable mCond;
		std::size_t mThreshold;		//需要等待的线程数
		std::size_t mCount;			//当前等待的线程数
		std::size_t mGeneration;	//区分不同同步点
	public:
		explicit barrier_cond(std::size_t iCount) :mThreshold(iCount), mCount(iCount), mGeneration(0) {

		}

		void Wait() {
			std::unique_lock<std::mutex> lLock{ mMutex };
			auto lGen = mGeneration;
			if (!--mCount) {
				mGeneration++;		//更新同步点
				mCount = mThreshold;//重置等待数
				mCond.notify_all();	//condition_variable通知其他线程
			}
			else {
				mCond.wait(lLock, [this, lGen] {return lGen != mGeneration; });
			}
		}
	};





	void run() {
		const std::size_t thread_count = 5;

		// barrier_spin 示例
		barrier_spin spin_barrier(thread_count);
		// barrier_cond 示例
		barrier_cond cond_barrier(thread_count);
		// std::barrier 示例（C++20）
		std::barrier std_barrier(thread_count, []() {
			std::cout << "Barrier action\n";
			});

		// 创建线程并使用不同的 barrier
		std::vector<std::thread> threads;
		for (std::size_t i = 0; i < thread_count; ++i) {
			threads.emplace_back([&, i]() {
				// 执行某些操作...
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
				// 然后等待所有线程到达这里
				spin_barrier.wait(); // 使用 barrier_spin
				cond_barrier.Wait(); // 使用 barrier_cond
				std_barrier.arrive_and_wait(); // 使用 std::barrier
				});
		}

		// 等待所有线程完成
		for (auto& t : threads) {
			t.join();
		}
	}
}






}