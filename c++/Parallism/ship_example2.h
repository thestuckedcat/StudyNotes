#pragma once
# include<thread>
# include<chrono>
# include<iostream>
# include<queue>

namespace ship_example_with_queue {
	/*
		��������
			����: �������ϰ�У�֮ǰ������������Ȼ���á�
			��������: ����Ҫ��������std::queue���͵ı�����һ����engine_workqueue����һ����clean_workqueue��
			�̱߳�ʾ: ����������๤��Ӧ�ø�����һ���̱߳�ʾ����������֮ǰ����ֻ��һ��������
			ѭ��ִ��: ��Щ�߳�Ӧ�ó������У�ͬʱ���һ����Ϊdone_flag�ı�־�����ӿ���̨���յ��û���������100��ʱ��Ӧ�����������־��ֹͣÿ��������
			���м��: ÿ���߳�Ӧ�����Ӧ�Ķ����Բ��ҹ��������߳̿��Խ��������ÿ����Ӧ�Ķ����С�
			����ִ��: ��������е��߳�����Ӧ�������ҵ��κ������Ӧ��ִ�и����������һ�롣
			������ʱ����: ���û���ҵ������߳�Ӧ���������롣
			*/
	void cleaner() {
		std::cout << "-----------------------Cleaning the board\n";
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));

	}

	void EngineCrew_fullspeedahead() {
		std::cout << "+++++++++++++++++++++++Full speed ahead\n";
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));

	}

	void EngineCrew_StopEngine() {
		std::cout << "=============================StopEngine\n";

	}
	void engine_worker(std::queue<int> &engine_work) {
		while (true) {
			if (engine_work.empty()) {
				printf("Engine work list is empty\n");
				std::this_thread::sleep_for(std::chrono::seconds(2));
			}
			else {
				int temp = engine_work.front();
				engine_work.pop();
				if (temp == 2)
					EngineCrew_fullspeedahead();
				else if (temp == 3)
					EngineCrew_StopEngine();
				else
				{
					printf("Engine worker find stop code\n");
					break;
				}
			}
		}
	}

	void clean_worker(std::queue<int>& clean_work) {
		while (true) {
			if (clean_work.empty()) {
				printf("Clean work is not exist\n");
				std::this_thread::sleep_for(std::chrono::seconds(2));
			}
			else {
				int temp = clean_work.front();
				clean_work.pop();
				if (temp == 100) {
					printf("Clean work find stop code\n");
					break;
				}
				else {
					cleaner();

				}
			}
		}
	}

	void captain() {
		bool done_flag = true;
		std::queue<int> engine_work, clean_work;
		std::thread thread_engine(engine_worker, std::ref(engine_work));
		std::thread thread_clean(clean_worker, std::ref(clean_work));
		thread_clean.detach();

		while (done_flag) {
			int command;
			std::cout << "�������ָ��" << std::endl;
			std::cin >> command;
			if (command == 100) {
				done_flag = false;
				engine_work.push(command);
				clean_work.push(command);
			}
			else if(command == 3 || command == 2) {
				engine_work.push(command);
			}
			else if (command == 1) {
				clean_work.push(command);
			}
			
		}
		thread_engine.join();


	}
}