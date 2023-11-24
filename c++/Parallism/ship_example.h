#pragma once
# include<iostream>
# include<thread>
# include<vector>
# include<chrono>
# include "call_thread_guard.h"



namespace ship_example {
	/*
	�������������ߣ�Captain������EngineCrew�����Լ�Cleaners��๤
	
	���������̱߳�ʾ

	�������Է����������
	���������Ա��࣬���Ǵ������ص�����������
	������������������ȫ��ǰ����ֹͣ�����������������ȵ����������Щ������ܼ���ִ����һ������

	Input��һ�����������������
	1����cleanning
	2����ȫ��ǰ��
	3����ֹͣ������
	100���˳�����
	�����������Ч��������Ҫ�ڿ���̨��ӡ

	ʹ��sleep��cout��ģ���������
	*/
	auto program_start = std::chrono::high_resolution_clock::now();
	void show_time() {
		auto now = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = now - program_start;
		std::cout << "Current timestamp: " << elapsed.count() << " seconds since program start." << std::endl;
	}

	void cleaner() {
		std::cout << "Cleaning the board\n";
		std::this_thread::sleep_for(std::chrono::milliseconds(5000));
		std::cout << "Cleaning complete\n";

	}

	void EngineCrew_fullspeedahead() {
		std::cout << "Full speed ahead\n";
		std::this_thread::sleep_for(std::chrono::milliseconds(5000));
		std::cout << "Slow down\n";

	}

	void EngineCrew_StopEngine() {
		std::cout << "StopEngine\n";

	}

	void captain(std::vector<int> Execution_code_list) {
		std::vector<std::thread> join_list;
		std::vector<std::thread> detach_list;
		for (auto code : Execution_code_list) {
			if (code == 100) break;

			switch (code) {
			case 1:
				detach_list.push_back(std::thread(cleaner));
				break;
			case 2:
				join_list.push_back(std::thread(EngineCrew_fullspeedahead));
				break;
			case 3:
				join_list.push_back(std::thread(EngineCrew_fullspeedahead));
				break;
			default:
				std::cout << "Invalid command.\n";
			}
		}
		for (auto &threadd : detach_list) {
			threadd.detach();
		}
		for (auto& threadadd : join_list) {
			threadadd.join();
			//show_time();
		}

		
	}
}