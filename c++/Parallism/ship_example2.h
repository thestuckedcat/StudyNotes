#pragma once
# include<thread>
# include<chrono>
# include<iostream>
# include<queue>

namespace ship_example_with_queue {
	/*
		任务描述
			命令: 在这个练习中，之前描述的命令仍然适用。
			工作队列: 您需要创建两个std::queue类型的变量，一个是engine_workqueue，另一个是clean_workqueue。
			线程表示: 发动机和清洁工作应该各自由一个线程表示，而不是像之前那样只用一个函数。
			循环执行: 这些线程应该持续运行，同时检查一个名为done_flag的标志。当从控制台接收到用户请求（输入100）时，应该设置这个标志以停止每个迭代。
			队列检查: 每个线程应检查相应的队列以查找工作。主线程可以将命令放入每个相应的队列中。
			命令执行: 如果运行中的线程在相应队列中找到任何命令，它应该执行该命令，并休眠一秒。
			无任务时休眠: 如果没有找到任务，线程应该休眠两秒。
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
			std::cout << "输入你的指令" << std::endl;
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