#pragma once
#include <chrono>
#include <iostream>
#include <thread>

namespace transfer_ownership {
	void foo() {
		//std::cout << "Thread ID " << std::this_thread::get_id() << " from foo\n";
		printf("Thread ID %d from foo\n", std::this_thread::get_id());
	}

	void bar() {
		//std::cout << "Thread ID " << std::this_thread::get_id() << " from bar\n";
		printf("Thread ID %d from bar\n", std::this_thread::get_id());
	}

	void run() {
		std::thread thread_1(foo);

		std::thread thread_2 = std::move(thread_1);

		thread_1 = std::thread(bar);
		//�������������ʽ���ƶ����ã���Ϊ�ұ�����ֵ�����Ǹ�ֵ������û����

		/*
		std::thread_3(foo);
		thread_1 = std::move(thread_3);//��һ��������throw һ��exception����Ϊthread1�й�����̣߳�����Ĳ���ʵ�������ڸ�������Ȩ����ת��
		*/
		thread_1.join();
		thread_2.join();
	}
}