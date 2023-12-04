#pragma once
# include <iostream>
# include <future>
# include <numeric>
# include <vector>


namespace accumulate_with_async {
	int MIN_ELEMENT_COUNT = 1000;

	template<typename iterator>
	int parallel_accumulate(iterator begin, iterator end) {
		long length = std::distance(begin, end);
		
		if (length <= MIN_ELEMENT_COUNT) {
			std::cout << "thread id is" << std::this_thread::get_id() << std::endl;
			return std::accumulate(begin, end, 0);
		}

		iterator mid = begin;
		std::advance(mid, (length + 1) / 2);

		// made the recursive call using std async task
		std::future<int> f1 = std::async(std::launch::deferred | std::launch::async, parallel_accumulate<iterator>, mid, end);
		// made the recursive call in current thread
		auto sum = parallel_accumulate<iterator>(begin, mid);
		return sum + f1.get();
	}

	void run() {
		std::vector<int> v(10000, 1);
		std::cout << "The sum is " << parallel_accumulate<std::vector<int>::iterator>(v.begin(), v.end()) << '\n';
	}
}