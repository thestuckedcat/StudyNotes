#pragma once

# include<iostream>
# include<thread>
# include<atomic>
# include<cassert>

namespace compare_exchange_example {
	void run(){
		std::atomic<int> x(20);

		int expected_value = 20;

		std::cout << "previous expected_value - " << expected_value << std::endl;
		bool return_val = x.is_lock_free()? x.compare_exchange_weak(expected_value, 6) : x.compare_exchange_strong(expected_value, 6);

		std::cout << "operation successful		- " << (return_val) << std::endl;
		std::cout << "current expeceted_value	- " << expected_value << std::endl;
		std::cout << "current x					- " << x.load() << std::endl;

	}
}


namespace atomic_pointer_example {
	void run_commom_function_example() {

		// common functions
		int values[20];

		for (int i = 0; i < 20; i++) {
			values[i] = i + 1;
		}

		std::atomic<int*> x_pointer = values;
		std::cout << "atomic integer pointer lock free ? " << (x_pointer.is_lock_free() ? "yes" : "no") << std::endl;

		int* y_pointer = values + 3;


		x_pointer.store(y_pointer);
		std::cout << "value referening to by pointer : " << *(x_pointer.load()) << std::endl;
		bool ret_val = x_pointer.compare_exchange_weak(y_pointer, values + 10);
		std::cout << "store operation successfull: " << (ret_val ? "yes" : "no") << std::endl;
		std::cout << "new value pointer by atomic pointer : " << *x_pointer << std::endl;
	}


	void run_operations() {
		int values[20];

		for (int i = 0; i < 20; i++) {
			values[i] = i + 1;
		}

		std::atomic<int*> x_pointer = values;

		std::cout << "1. After initialization value pointed by atomic pointer - " << *x_pointer << std::endl;

		//fetch add +=
		int* prev_pointer_val_1 = x_pointer.fetch_add(12);
		std::cout << "2. After fetch add previous value pointed by atomic pointer - " << *prev_pointer_val_1 << std::endl;
		std::cout << "2. After fetch add new value pointed by atomic pointer - " << *x_pointer << std::endl;

		//fetch_sub -=
		int* prev_pointer_val_2 = x_pointer.fetch_sub(3);
		std::cout << "3. After fetch sub previous value pointer by atomic pointer - " << *prev_pointer_val_2 << std::endl;
		std::cout << " 3. After fetch sub new value pointed by atomic pointer - " << *x_pointer << std::endl;

		//++ operator
		x_pointer++;
		std::cout << "4. After post increment value pointed by atomic pointer - " << *x_pointer << std::endl;


		//-- operator
		x_pointer--;
		std::cout << "5. After post decrement value pointed by atomic pointer - " << *x_pointer << std::endl;

	}



}


namespace memory_model_example {
	std::atomic<bool> x, y;
	std::atomic<int> z;

	void write_x() {
		//set x to true
		x = true;
	}

	void write_y() {
		//set y to true
		y = true;
	}

	void read_x_then_y() {
		//loop until x is true
		while (!x.load(std::memory_order_seq_cst));

		//check wether y is true
		if (y.load(std::memory_order_seq_cst)) {
			z++;
		}
	}

	void read_y_then_x() {
		//loop until y is true
		while (!y.load(std::memory_order_seq_cst));

		//check wether x is true
		if (x.load(std::memory_order_seq_cst)) {
			z++;
		}
	}

	void run_code() {
		x = false;
		y = false;
		z = 0;

		
		std::thread thread_c(read_x_then_y);
		std::thread thread_d(read_y_then_x);
		std::thread thread_a(write_x);
		std::thread thread_b(write_y);

		thread_a.join();
		thread_b.join();
		thread_c.join();
		thread_d.join();
		std::cout << z.load() << std::endl;
		assert(z != 0);


	}
}