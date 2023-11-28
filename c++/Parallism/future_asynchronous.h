#pragma once
# include<iostream>
# include<future>

namespace future_asynchronous {
	bool wait_the_data_downloading() {
		//这里是一个传输过程，不过我们不能用sleep_for模拟，因为我们没有指定async的launch模式，就假装需要5秒
		return true;
	}

	void do_other_calculations() {
		std::cout << "doing other stuff" << std::endl;
	}

	void run() {
		//Aquire the future associated with that particular asynchronous
		std::future<bool> the_answer_future = std::async(wait_the_data_downloading);
		do_other_calculations();
		std::cout << "The downloading is complete?  " << the_answer_future.get() << std::endl;

	}
}
