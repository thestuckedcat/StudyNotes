#pragma once

# include<iostream>
# include<list>
# include<algorithm>
# include<future>
# include<chrono>
# include<random>
namespace parallel_quick_sort1 {
	//顺序执行
	template<typename T>
	std::list<T> sequential_quick_sort(std::list<T> input) {
		
		if (input.size() < 2)
		{
			return input;
		}
		//select the pivot value:repoint the first element from input to result 
		
		std::list<T> result;
		result.splice(result.begin(), input, input.begin());//splice(iterator pos, list &other, iterator it);  Transfers the element pointed to by it from other into *this. The element is inserted before the element pointed to by pos.
		T pivot = *result.begin();		//pick first element as pivot

		//arrange the input array
		auto divide_point = std::partition(input.begin(), input.end(), [&](T const& t) {return t < pivot; });//partition(first,last,p); Reorders the elements in the range [first, last) in such a way that all elements for which the predicate p returns true precede the elements for which predicate p returns false. Relative order of the elements is not preserved. Return iterator to the first element of second group(*devide_point >= pivot)
		
		// call the sequential_quick_sort recursively
		std::list<T> lower_list;
		lower_list.splice(lower_list.end(), input, input.begin(), divide_point);//lower part is in lower_list and upperpart is in input list

		auto new_lower(sequential_quick_sort<T>(std::move(lower_list)));
		auto new_upper(sequential_quick_sort<T>(std::move(input)));

		// arranging the result list
		result.splice(result.begin(), new_lower);//Transfers all elements from other into *this. The elements are inserted before the element pointed to by pos. The container other becomes empty after the operation.
		result.splice(result.end(), new_upper);

		return result;
	}






	//并行版本
	template<typename T>
	std::list<T> parallel_quick_sort(std::list<T> input) {
		//std::cout << std::this_thread::get_id() << std::endl;
		if (input.size() < 2) {
			return input;
		}
		//move first element in the list to result list and take it as pivot value
		std::list<T> result;
		result.splice(result.begin(), input, input.begin());
		T pivot = *result.begin();

		//partition the input array
		auto divide_point = std::partition(input.begin(), input.end(), [&](T const& t) {return t < pivot; });

		//move lower part of the list to separate list so that we can make recursive call
		std::list<T> lower_list;
		lower_list.splice(lower_list.end(), input, input.begin(), divide_point);

		auto new_lower(parallel_quick_sort<T>(std::move(lower_list)));
		//apply async to recrusive
		std::future<std::list<T>> new_upper_future(std::async(std::launch::async|std::launch::deferred, &parallel_quick_sort<T>, std::move(input)));

		result.splice(result.begin(), new_lower);
		result.splice(result.end(), new_upper_future.get());

		return result;
		
	}
	//std::list<double> parallel_quick_sort(std::list<double> input) 
	//{
	//	//std::cout << std::this_thread::get_id() << std::endl;
	//	if (input.size() < 2) {
	//		return input;
	//	}
	//	//move first element in the list to result list and take it as pivot value
	//	std::list<double> result;
	//	result.splice(result.begin(), input, input.begin());
	//	double pivot = *result.begin();

	//	//partition the input array
	//	auto divide_point = std::partition(input.begin(), input.end(), [&](double const& t) {return t < pivot; });

	//	//move lower part of the list to separate list so that we can make recursive call
	//	std::list<double> lower_list;
	//	lower_list.splice(lower_list.end(), input, input.begin(), divide_point);

	//	auto new_lower(parallel_quick_sort(std::move(lower_list)));
	//	//将其中一个递归函数调用作为async task
	//    std::future<std::list<double>> new_upper_future(std::async(&parallel_quick_sort, std::move(input)));


	//	result.splice(result.begin(), new_lower);
	//	result.splice(result.end(), new_upper_future.get());

	//	return result;

	//}


	//std::list<double> my_parallel_quick_sort(std::list<double> input)
	//{
	//	if (input.size() <= 10000000) {
	//		return sequential_quick_sort<double>(input);
	//	}
	//	//std::cout << std::this_thread::get_id() << std::endl;
	//	if (input.size() < 2) {
	//		return input;
	//	}
	//	//move first element in the list to result list and take it as pivot value
	//	std::list<double> result;
	//	result.splice(result.begin(), input, input.begin());
	//	double pivot = *result.begin();

	//	//partition the input array
	//	auto divide_point = std::partition(input.begin(), input.end(), [&](double const& t) {return t < pivot; });

	//	//move lower part of the list to separate list so that we can make recursive call
	//	std::list<double> lower_list;
	//	lower_list.splice(lower_list.end(), input, input.begin(), divide_point);

	//	auto new_lower(parallel_quick_sort(std::move(lower_list)));
	//	//将其中一个递归函数调用作为async task
	//	std::future<std::list<double>> new_upper_future(std::async(&parallel_quick_sort<double>, std::move(input)));


	//	result.splice(result.begin(), new_lower);
	//	result.splice(result.end(), new_upper_future.get());

	//	return result;

	//}


	




	template<typename T>
	void print_runtime(const char* const tag, const std::list<T> &sorted, std::chrono::high_resolution_clock::time_point start_time, std::chrono::high_resolution_clock::time_point end_time) {
		printf("%s : Lowest : %lf Highest: %lf Time : %lf\n", tag, sorted.front(), sorted.back(), std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_time - start_time).count());
	}
	void run(size_t test_Size, int iterationCount) {
		std::random_device rd;

		//generate some random doubles
		printf("Testing with %zu doubles...\n", test_Size);
		std::list<double> doubles(test_Size);
		for (auto& d : doubles) {
			d = static_cast<double>(rd());
		}

		//sequential
		for (int i = 0; i < iterationCount; i++) {
			std::list<double> sorted;
			const std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
			sorted = sequential_quick_sort<double>(doubles);
			const std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
			print_runtime<double>("Sequential quick sort", sorted, startTime, endTime);
		}

		//parallel
		for (int i = 0; i < iterationCount; i++)
		{
			std::list<double> sorted;
			const std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
			//sorted = parallel_quick_sort(doubles);
			sorted = parallel_quick_sort<double>(doubles);
			const std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
			print_runtime<double>("Parallel quick sort", sorted, startTime, endTime);
		}

		//my parallel
		//for (int i = 0; i < iterationCount; i++)
		//{
		//	std::list<double> sorted;
		//	const std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
		//	sorted = my_parallel_quick_sort(doubles);
		//	//sorted = parallel_quick_sort<double>(doubles);
		//	const std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
		//	print_runtime<double>("My Parallel quick sort", sorted, startTime, endTime);
		//}

	}


}