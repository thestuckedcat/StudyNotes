# include <atomic>
# include <vector>
# include <iostream>
# include "thread_safe_queue.h"
# include <functional>
# include <algorithm>
# include <list>
# include <memory>
# include <queue>
# include <future>
# include <numeric>
# include "thread_pool_local_queue.h"


namespace my_thread_pool_with_local_queue {
	thread_local std::unique_ptr<std::queue<function_wrapper>> thread_pool::local_work_queue;
	void run_quick_sort() {
		const int size = 800;
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