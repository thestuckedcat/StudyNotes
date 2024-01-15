# include "thread_pool_work_stealing.h"

namespace thread_pool_with_work_stealing {
	thread_local work_stealing_queue* thread_pool_with_work_steal::local_work_queue;
	thread_local unsigned thread_pool_with_work_steal::my_index;

	void run_quick_sort() {
		const int size = 50;
		std::list<int> my_array;
		srand(0);

		for (size_t i = 0; i < size; i++) {
			my_array.push_back(rand());
		}

		my_array = parallel_quick_sort(my_array);
		for (size_t i = 0; i < size; i++) {
			std::cout << my_array.front() << std::endl;
			my_array.pop_front();
		}
	}
}