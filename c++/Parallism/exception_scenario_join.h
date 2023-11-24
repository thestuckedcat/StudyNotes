#pragma once

# include<iostream>
# include<vector>
# include<thread>
# include "call_thread_guard.h"

namespace exception_scenario {
	void foo(){
		std::cout << "This is from foo" << std::endl;
	}

	void other_operations() {
		std::cout << "An error will be thrown\n";
		throw std::runtime_error("This is a runtime error");
	}

	void run() {
		std::thread foo_thread(foo);
		thread_guard tg(foo_thread);
		try {
			other_operations();
		}
		catch (...) {

		}
	}

}