#pragma once
# include<iostream>
# include<thread>
# include<vector>
# include<chrono>
# include "call_thread_guard.h"



namespace ship_example {
	/*
	考虑三个参与者，Captain船长，EngineCrew机组以及Cleaners清洁工
	
	船长由主线程表示

	船长可以发出三个命令：
	命令清洁人员清洁，但是船长不必等这个命令完成
	接下来的两个命令是全速前进和停止发动机命令，船长必须等到机组完成这些命令才能继续执行下一个命令

	Input是一个整数代表上述命令：
	1代表cleanning
	2代表全速前进
	3代表停止发动机
	100是退出程序
	其他编号是无效订单，需要在控制台打印

	使用sleep和cout来模拟这个过程
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