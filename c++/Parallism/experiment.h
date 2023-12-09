#pragma once
# include <thread>
# include <iostream>
# include <vector>

namespace experimental_code {
	class Task {
		std::vector<std::thread> threads;
		int numThreads{ 0 };

	public:
		Task() = default;

		//start in a new thread
		template<typename Functiontype>
		void start(Functiontype f) {
			threads.push_back( std::thread{ std::move(f) });
			numThreads ++ ;
		}

		//destructor
		~Task() {
			for (auto& t : threads) {
				t.join();
			}
		}

		Task(Task&& _new) noexcept
			:numThreads{_new.numThreads}, threads(std::move(_new.threads))
		{
			_new.numThreads = 0;
		}

		Task &operator=(Task&& _new) noexcept{
			if (this != &_new) {
				numThreads = _new.numThreads;
				threads = std::move(_new.threads);
				_new.numThreads = 0;
			}
		}

		Task(const Task&) = delete;
		Task& operator=(const Task&) = delete;
	};
}
