#pragma once

# include<iostream>
# include<future>
# include<numeric>
# include<thread>
# include<functional>


namespace package_task_example {
	/*����������У�����ʹ��package_task������add function*/
	int add(int x, int y) {
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		std::cout << "add function runs in : " << std::this_thread::get_id() << std::endl;
		return x + y;
	}

	void task_normal() {
		std::packaged_task<int(int, int)> task_1(add);
		std::future<int> future_1 = task_1.get_future(); // acquire the associated future with that package task 
		
		//����async�ڹ������Զ����У�������Ҫ��ʽ���ã�����future get���޷�����
		
		task_1(7, 8);									// �����������

		//���������£���������������ڵ�ǰ�߳��а�˳������(ͬ��ִ�У������ַ�ʽչ����package�������κ��߳���ִ��

		std::cout << "task normal - " << future_1.get() << "\n";
	}

	//�����Ҫ�첽����������Ҫ����һ���̲߳��������񴫵ݸ����߳�
	void task_thread() {
		std::packaged_task<int(int, int)> task_1(add);
		std::future<int> future_1 = task_1.get_future();

		std::thread thread_1(std::move(task_1), 5, 6);//�������ǾͿ����첽�������������package_task����copyable�ģ���˱���Ҫmove
		thread_1.detach();							//�첽������

		std::cout << "task thread - " << future_1.get() << "\n";
	}

	void run() {
		task_thread();
		task_normal();
		std::cout << "main thread id : " << std::this_thread::get_id() << std::endl;
	}
}