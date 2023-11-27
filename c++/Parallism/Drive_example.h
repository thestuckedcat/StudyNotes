#pragma once
# include<iostream>
# include<mutex>
# include<thread>
# include<string>
# include<chrono>

# include<condition_variable>

namespace drive_example {
	bool have_i_arrived = false;
	int distance_my_destination = 10;	//距离目标的路途
	int distance_covered = 0;			//当前已经走过的路途

	bool keep_driving() {
		while (true) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			distance_covered++;
		}
		return false;
	}


	void keep_awake_all_night() {
		while (distance_covered < distance_my_destination) {//这个while的判断语句模拟了我们向司机询问的这一行为，这个判断会花费一定的处理器时间
			std::cout << "keep check, whether i am there\n";
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));

		}
		std::cout << "keep_awake: finally i am there, distance covered = " << distance_covered << std::endl;
	}

	void set_the_alarm_and_take_a_nap() {
		if (distance_covered < distance_my_destination) {
			std::cout << "let me take a nap\n";
			std::this_thread::sleep_for(std::chrono::milliseconds(10000));
		}
		if (distance_covered < distance_my_destination)
			std::cout << "alarm: oh shit I wake up early, distance_covered = " << distance_covered << std::endl;
		else if (distance_covered == distance_my_destination)
			std::cout << "alarm: exactly the time I arrived, distance_covered = " << distance_covered << std::endl;
		else
			std::cout << "alarm: oh shit I pass my distination, distance_covered = " << distance_covered << std::endl;
	}


	void run() {
		std::thread driving(keep_driving);
		std::thread awake(keep_awake_all_night);
		std::thread set_alarm(set_the_alarm_and_take_a_nap);


		driving.join();
		awake.join();
		set_alarm.join();
	}

	//----------------------------------------------example2-----------------------------------------
	int total_distance = 10;
    int distance_coveredd = 0;
	std::condition_variable cv;
	std::mutex m;

	void keep_moving() {
		while (true) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			distance_coveredd++;

			if (distance_coveredd == total_distance)
				cv.notify_all();
		}
	}

	void ask_driver_to_wake_u_up_at_right_time() {
		std::unique_lock<std::mutex> ul(m);
		cv.wait(ul, [] {return distance_coveredd == total_distance; });//get ownership of the mutex associated with the unique lock and check whether the condition specified in the lambda expression is true
		//It will not be true in this first moment
		//Therefore condition variable will unlock the associated mutex and make passenger thread sleep until it gets notified
		// Condition variable should able to called lock and unlock on the associated mutex, that is the reason why we are using unique locks here
		//we cannot use lock guard objects here because it doesnot provide this flexibility
		// Once the notify call from the driver thread that notification will cause any thread wait on particular condition variable to wake up
		//Now in this case, our passenger thread will wake up due to this notification 
		// it will first lock the associated mutex and check the condition in the lambda expression
		//lambda return true ,therefore passenger will be allowed to proceed to the next statement and printed out
		// 为什么会有一个判断语句呢？因为这一个线程被不止能够被notify唤醒，还能被操作系统唤醒，毕竟操作系统不会特地分别你线程属性，因此如果被操作系统唤醒，它仍然会check这个条件，然后再度睡眠
		std::cout << "finally i am there, distance_covered = " << distance_coveredd << std::endl;
	}

	void run_2() {
		std::thread driver_thread(keep_moving);
		std::thread passenger_thread(ask_driver_to_wake_u_up_at_right_time);

		passenger_thread.join();
		driver_thread.join();
	}

}