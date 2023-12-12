#pragma once


# include <thread>
# include <atomic>
# include <condition_variable>
# include <barrier>
# include <iostream>
# include <vector>

namespace barrier_implemention {
	/*
	ʹ�������ȴ�����ʵ��barrier
	spin the threads until the required number of threads reached the barrier
	*/

	class barrier_spin {

		unsigned const thresh_hold; //��Ҫ�ȴ����߳���
		std::atomic<unsigned> count;//��ǰ�ȴ����߳���
		std::atomic<unsigned> generation;//���ڶ��ͬ��ʱ�����ֲ�ͬ��ͬ���㣬���ֻ��һ��ͬ����Ͳ���Ҫ

	public:
		explicit barrier_spin(unsigned _thresh_hold) :thresh_hold(_thresh_hold), count(_thresh_hold), generation(0) {

		}

		void wait() {
			//�ȴ��߳���������Ҫ��
			//ÿ���̶߳������
			unsigned const my_generation = generation;

			if (!--count) {
				// ���һ��������߳����ü����������� generation
				count = thresh_hold;
				++generation;
			}
			else {
				//�������߳���æ�ȴ�
				while (generation == my_generation)
					std::this_thread::yield();
				/*
				�ó�ִ��Ȩ������ std::this_thread::yield() ���õ�ǰִ�е��̸߳��߲���ϵͳ��Ը�����ʣ���ʱ��Ƭ������˲���ϵͳ����ȥ���������߳����С����ǣ���������һ�������顱�򡰰�ʾ��������ϵͳ���ܻ�����������

				����æ�ȴ���busy-waiting����Ӱ�죺��ĳЩ�㷨�У��߳̿��ܻ����æ�ȴ�״̬�������ϼ��ĳ�������Ƿ������������ʵ�ʵ����ù���������������£�ʹ�� std::this_thread::yield() ���Լ��ٶ�CPU��Դ���˷ѣ���Ϊ���������ϵͳ��æ�ȴ����̺߳������߳�֮�����Ч���л���

				��߶��̳߳������Ӧ�Ժ͹�ƽ�ԣ���һЩ��Ҫƽ�����߳�ִ�еĳ����У�ʹ�� std::this_thread::yield() ���԰�����߳������Ӧ�Ժ����еĹ�ƽ�ԡ�
				
				�����ѭ���У�
				�̻߳᲻�ϼ�� generation �Ƿ���� my_generation������������������ͻ���� yield������һ�������ϵͳ�������źţ���ʾ�߳�Ը���ó���ʣ���ʱ��Ƭ��
				�������ϵͳ���ܣ���ô����̻߳ᱻ��ʱdeactive��ֱ������ϵͳ�ٴν������Ϊ�״̬������ȴ�ʱ����ܻ�������ͬ��ȡ���ڲ���ϵͳ�ĵ��Ȳ��Ժ�ϵͳ�����е���������
				������ϵͳ�����ٴε��ȸ��߳�ʱ���߳̽��� yield ����֮��Ĵ������ִ�С�
				*/
			}
		}
	};


	/*
	ʹ�ù������
	*/
	class barrier_cond {

		std::mutex mMutex;
		std::condition_variable mCond;
		std::size_t mThreshold;		//��Ҫ�ȴ����߳���
		std::size_t mCount;			//��ǰ�ȴ����߳���
		std::size_t mGeneration;	//���ֲ�ͬͬ����
	public:
		explicit barrier_cond(std::size_t iCount) :mThreshold(iCount), mCount(iCount), mGeneration(0) {

		}

		void Wait() {
			std::unique_lock<std::mutex> lLock{ mMutex };
			auto lGen = mGeneration;
			if (!--mCount) {
				mGeneration++;		//����ͬ����
				mCount = mThreshold;//���õȴ���
				mCond.notify_all();	//condition_variable֪ͨ�����߳�
			}
			else {
				mCond.wait(lLock, [this, lGen] {return lGen != mGeneration; });
			}
		}
	};





	void run() {
		const std::size_t thread_count = 5;

		// barrier_spin ʾ��
		barrier_spin spin_barrier(thread_count);
		// barrier_cond ʾ��
		barrier_cond cond_barrier(thread_count);
		// std::barrier ʾ����C++20��
		std::barrier std_barrier(thread_count, []() {
			std::cout << "Barrier action\n";
			});

		// �����̲߳�ʹ�ò�ͬ�� barrier
		std::vector<std::thread> threads;
		for (std::size_t i = 0; i < thread_count; ++i) {
			threads.emplace_back([&, i]() {
				// ִ��ĳЩ����...
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
				// Ȼ��ȴ������̵߳�������
				spin_barrier.wait(); // ʹ�� barrier_spin
				cond_barrier.Wait(); // ʹ�� barrier_cond
				std_barrier.arrive_and_wait(); // ʹ�� std::barrier
				});
		}

		// �ȴ������߳����
		for (auto& t : threads) {
			t.join();
		}
	}
}






}