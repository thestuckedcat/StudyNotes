#pragma once
# include <iostream>
# include <mutex>
# include <thread>
# include <string>
# include <chrono>

namespace dead_lock_example {

	//--------------------------------example 1-------------------------------------------
	class bank_account {
		double balance;
		std::string name;
		std::mutex m;
	public:
		bank_account() {};

		bank_account(double _balance, std::string _name) :balance(_balance), name(_name) {}
		
		bank_account(bank_account& const) = delete; //禁用拷贝构造函数

		bank_account& operator=(bank_account& const) = delete;

		void withdraw(double amount) {
			std::lock_guard<std::mutex> lg(m);
			balance += amount;
		}

		void deposite(double amount) {
			std::lock_guard<std::mutex> lg(m);
			balance += amount;
		}

		void transfer(bank_account& from, bank_account& to, double amount) {
			/*这个函数将from账户的钱转到to账户*/
			std::lock_guard<std::mutex> lg_1(from.m);
			std::cout << "lock for" << from.name << " account acuqire by " << std::this_thread::get_id() << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(1));

			std::cout << "waiting to acquire lock for " << to.name << " account by " << std::this_thread::get_id() << std::endl;
			std::lock_guard<std::mutex> lg_2(to.m);

			from.balance -= amount;
			to.balance += amount;
			std::cout << "transfer to - " << to.name << " from - " << from.name << "end \n";


		}

	};

	void run_1() {
		bank_account account;
		bank_account account_1(1000, "james");
		bank_account account_2(2000, "Mathew");

		std::thread thread_1(&bank_account::transfer, &account, std::ref(account_1), std::ref(account_2), 500);
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		std::thread thread_2(&bank_account::transfer, &account, std::ref(account_2), std::ref(account_1), 200);

		thread_1.join();
		thread_2.join();
	}


	//----------------------------------example 2---------------------------------------
	std::mutex m1;
	std::mutex m2;

	void m1_first_m2_second() {
		/*首先获取M1相关锁，然后睡眠，然后获取M2相关锁*/
		std::lock_guard<std::mutex> lg1(m1);
		std::this_thread::sleep_for(std::chrono::seconds(1));
		printf("Thread %d has acquired lock for m1 mutex\n", std::this_thread::get_id());
		
		std::lock_guard<std::mutex> lg2(m2);
		printf("Thread %d has acquired lock for m2 mutex\n", std::this_thread::get_id());
		
	}

	void m2_first_m1_second() {
		/*首先获取M1相关锁，然后睡眠，然后获取M2相关锁*/
		std::lock_guard<std::mutex> lg1(m2);
		std::this_thread::sleep_for(std::chrono::seconds(1));

		printf("Thread %d has acquired lock for m2 mutex\n", std::this_thread::get_id());
		std::lock_guard<std::mutex> lg2(m1);

		printf("Thread %d has acquired lock for m1 mutex\n", std::this_thread::get_id());
	}
	void run_2() {
		std::thread thread_1(m1_first_m2_second);
		std::thread thread_2(m2_first_m1_second);

		thread_1.join();
		thread_2.join();
	}




}