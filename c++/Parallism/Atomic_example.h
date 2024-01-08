#pragma once

# include<iostream>
# include<thread>
# include<atomic>
# include<cassert>
# include<memory>


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


namespace relaxed_order {
	std::atomic<bool> x, y;
	std::atomic<int> z;

	void write_x_then_y() {
		x.store(true, std::memory_order_relaxed);
		y.store(true, std::memory_order_relaxed);

	}

	void read_y_then_x() {
		while (!y.load(std::memory_order_relaxed)) {
			if (x.load(std::memory_order_relaxed)) {
				z++;
			}
		}
	}

	void run_code() {
		x = false;
		y = false;
		z = 0;

		std::thread writer_thread(write_x_then_y);
		std::thread reader_thread(read_y_then_x);
		
		
		writer_thread.join();
		reader_thread.join();

		assert(z != 0);

	}
}


namespace release_acquire_1 {
	std::atomic<bool> x, y;
	std::atomic<int> z;

	void write_x_then_y() {
		x.store(true, std::memory_order_relaxed);
		y.store(true, std::memory_order_release);
	}

	void read_y_then_x() {
		while (!y.load(std::memory_order_acquire));
		if (x.load(std::memory_order_relaxed)) {
			z++;
		}
	}

	void run_code() {
		x = false;
		y = false;
		z = 0;

		std::thread writer_thread(write_x_then_y);
		std::thread reader_thread(read_y_then_x);

		writer_thread.join();
		reader_thread.join();

		assert(z != 0);
	}
}

namespace release_acquire_2 {
	std::atomic<bool> x, y;
	std::atomic<int> z;

	void write_x(){
		x.store(true, std::memory_order_release);
	}

	void write_y() {
		y.store(true, std::memory_order_release);
	}

	void read_x_then_y() {
		while (!x.load(std::memory_order_acquire));

		if (y.load(std::memory_order_acquire)) {
			z++;
		}
	}

	void read_y_then_x() {
		while (!y.load(std::memory_order_acquire));

		if (x.load(std::memory_order_acquire)) {
			z++;
		}
	}

	void run() {
		x = false;
		y = false;
		z = 0;

		std::thread thread_a(write_x);
		std::thread thread_b(write_y);
		std::thread thread_c(read_x_then_y);
		std::thread thread_d(read_y_then_x);

		thread_a.join();
		thread_b.join();
		thread_c.join();
		thread_d.join();

		assert(z != 0);
	}
}


namespace transitive_synchronization {
	std::atomic<int> data[5];
	std::atomic<bool> sync1(false), sync2(false);

	void thread_1_func() {
		data[0].store(42, std::memory_order_relaxed);
		data[1].store(45, std::memory_order_relaxed);
		data[2].store(47, std::memory_order_relaxed);
		data[3].store(49, std::memory_order_relaxed);
		data[4].store(56, std::memory_order_relaxed);
		sync1.store(true, std::memory_order_release);

	
	}


	void thread_2_func() {
		// 为thread1,thread3施加synchronization
		while (!sync1.load(std::memory_order_acquire));
		sync2.store(true, std::memory_order_release);
	}

	void thread_3_func() {
		while (!sync2.load(std::memory_order_acquire));
		assert(data[0].load(std::memory_order_relaxed) == 42);
		assert(data[1].load(std::memory_order_relaxed) == 45);
		assert(data[2].load(std::memory_order_relaxed) == 47);
		assert(data[3].load(std::memory_order_relaxed) == 49);
		assert(data[4].load(std::memory_order_relaxed) == 56);
	}

	void run_code(){
		std::thread thread_1(thread_1_func);
		std::thread thread_2(thread_2_func);
		std::thread thread_3(thread_3_func);


		thread_1.join();
		thread_2.join();
		thread_3.join();

	}
}


namespace date_dependency_synchronization {
	struct X {
		int i;
		std::string s;

	};

	std::atomic<X*>p;
	std::atomic<int> a;

	void create_x() {
		X* x = new X;
		x->i = 42;
		x->s = "hello";

		a.store(20, std::memory_order_relaxed);
		p.store(x, std::memory_order_release);

	}

	void use_x() {
		X* x = nullptr;//不然会报错使用未初始化的局部变量。
		while (!(x == p.load(std::memory_order_consume)));//synchronize only for p, no guarantee for a,当然如果用acquire，鉴于他们之间有happens-before的关系，这个时候是能保证a已经被存储的
		assert(x->i == 42);
		assert(x->s == "hello");
		assert(a.load(std::memory_order_relaxed) == 20);
	}

	void run_code() {
		std::thread thread_create(create_x);
		std::thread thread_use(use_x);


		thread_create.join();
		thread_use.join();
	}

}

# include <queue>
# include<chrono>

namespace Release_sequence {
	std::atomic<int> count;
	std::queue<int> data_queue;

	int max_cout = 20;
	void writer_queue() {
		for (size_t i = 0; i < 20; i++) {
			data_queue.push(i);
		}
		count.store(20, std::memory_order_release);
	}

	void reader_queue() {
		while (true) {
			int item_index = 0;
			if (!(item_index = count.fetch_sub(1, std::memory_order_acquire) <= 0)) {//fetch sub是一个read modify write operation
				//wait for items
				std::this_thread::sleep_for(std::chrono::milliseconds(500));
				continue;
			}
			//process the item
		}
	}

	void run_code(){
		std::thread writer_thread(writer_queue);
		std::thread reader_thread1(reader_queue);
		std::thread reader_thread2(reader_queue);

		writer_thread.join();
		reader_thread1.join();
		reader_thread2.join();
	}

}


namespace my_spinlock_mutex {
	class spinlock_mutex {
		std::atomic_flag flag = ATOMIC_FLAG_INIT;

	public:
		spinlock_mutex() {}

		void lock() {
			while (flag.test_and_set(std::memory_order_acquire));
		}

		void unlock() {
			flag.clear(std::memory_order_release);
		}
	};



	//演示如何使用
	spinlock_mutex mutex;
	void func() {
		std::lock_guard<spinlock_mutex> lg(mutex);//any lock has both lock and unlock function can be used with guard object
		std::cout << std::this_thread::get_id() << " hello " << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(5000));
	}

	void run_code() {
		std::thread thread1(func);
		std::thread thread2(func);


		thread1.join();
		thread2.join();
	}
}