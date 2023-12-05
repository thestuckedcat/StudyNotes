#pragma once
# include <future>
# include <numeric>
# include <vector>
# include <iostream>
# include <chrono>
# include <algorithm>
namespace parallel_prefix_sum {

	class join_threads {
		std::vector<std::thread> &threads;

	public:
		explicit join_threads(std::vector<std::thread> &_threads):threads{_threads}{}

		~join_threads() {
			for (auto& t : threads) {
				if (t.joinable()) {
					t.join();
				}
			}
		}
	};

	template<typename Func>
	void print_function_run_time(Func callable) {
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		callable();
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		printf("Time %lf ms \n", std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count());
	}


	template<typename Iterator, typename OutIterator>
	void sequential_partial_sum(Iterator const first, Iterator const last, OutIterator y) {
		unsigned long length = std::distance(first, last);

		y[0] = first[0];

		for (size_t i = 1; i < length; i++) {
			y[i] = first[i] + y[i - 1];
		}
	}


	template<typename Iterator>
	void parallel_partial_sum(Iterator first, Iterator last) {
		/*
		typename的作用之一
		在模板代码内指定依赖类型：
		当在模板代码中引用一个依赖于模板参数的类型时，需要使用 typename。
		这是因为在模板实例化之前，编译器可能无法确定某个名称是否表示一个类型。
		例如，在 typename MyTemplate<T>::SubType x; 中，typename 用于告诉编译器 MyTemplate<T>::SubType 是一个类型
		在本例子中，因为我们需要使用向量化的prmomise和future，其返回值与未定义的Iterator相关，因此我们需要使用typename Iterator::value_type来指定
		*/
		typedef typename Iterator::value_type value_type;

		struct process_chunk {
			void operator()(Iterator begin, Iterator last, std::future<value_type>* previous_end_value, std::promise<value_type>* end_value){
				try {
					Iterator end = last;
					++end;
					std::partial_sum(begin, end, begin);
					if (previous_end_value != nullptr) {
						//this is not the first thread
						auto addend = previous_end_value->get();
						*last += addend;
						if (end_value) {
							//not the last block
							end_value->set_value(*last);
						}
						std::for_each(begin, last, [addend](value_type& item) {
							item += addend;
							});
					}
					else if (end_value) {
						//this is the first thread
						end_value->set_value(*last);
					}
				}
				catch(...)
				{
					if (end_value) {
						//promise is available，即为不是最后一个block
						end_value->set_exception(std::current_exception());

					}
					else {
						//final block -main thread is the one process the final block
						throw;
					}

				}

			}
		};

		unsigned long const length = std::distance(first, last);
		if (!length)
			return;

		// 计算最优线程数
		unsigned long const min_per_thread = 25;
		unsigned long const max_threads = (length + min_per_thread - 1) / min_per_thread;
		unsigned long const hardware_threads = std::thread::hardware_concurrency();
		unsigned long const num_threads = std::min(hardware_threads != 0 ? hardware_threads : max_threads, max_threads);
		unsigned long const block_size = length / num_threads;

		//会用到的向量
		std::vector<std::thread> threads(num_threads - 1);
		std::vector<std::promise<typename Iterator::value_type>> end_values(num_threads - 1);	//向下一个thread发送累加值 ,这里展示了没有typedef应该怎么写
		std::vector<std::future<value_type>> previous_end_values;			//接收上一个thread的累加值
		previous_end_values.reserve(num_threads - 1);
		/*
		这里使用reserve而非直接构造，原因如下

		1.reserve 函数不会调用任何元素的构造函数。reserve 仅用于调整容器的容量，也就是它可以存储的元素数量，但不改变容器的实际大小（即容器中元素的数量）。
		当您调用 reserve 时，std::vector 会分配足够的内存以存储指定数量的元素，但不会创建或初始化这些元素。
		这意味着 reserve 后，容器的 size() 仍然保持不变，只是其 capacity() 增加了。

		在这里，如果我们直接使用previous_end_values(num_threads-1),我们后面就应该使用previous_end_values[flag++] = end_values[i].get_future();此处是移动构造
		这两个版本，现在的版本发生了申请空间和一次push
		直接构造版本发生了申请空间，初始化构造开销以及移动赋值构造，这些开销在数据量大时不容忽视
		因此，在这种元素逐步添加的场景中，使用reserve更加合理

		2. 对比另一个resize

		不同于 reserve，resize 函数会改变 std::vector 的实际大小，即容器中元素的数量。
		当你调用 resize 时，如果新的大小大于当前大小，std::vector 会添加足够数量的元素以达到指定的大小。
		这些新添加的元素会被默认构造，或者如果 resize 提供了一个值作为第二个参数，新元素将被初始化为该值。
		
		*/

		//主操作
		join_threads joiner(threads);

		Iterator block_start = first;
		for (unsigned long i = 0; i < (num_threads - 1); i++)
		{
			Iterator block_last = block_start;
			std::advance(block_last, block_size - 1);

			threads[i] = std::thread(process_chunk(), block_start, block_last, (i != 0) ? &previous_end_values[i - 1] : nullptr, &end_values[i]);

			/*这里有第二种写法,采取引用传递，稍显麻烦，这也体现了这里使用指针传递的方便之处
			std::future<value_type> default_future;
			threads[i] = std::thread(process_chunk(), block_start, block_last, (i != 0) ? std::ref(previous_end_values[i - 1]) : default_future, std::ref(end_values[i]));
			对应的函数
			void operator()(Iterator begin, Iterator last, std::future<value_type> &previous_end_value, std::promise<value_type> &end_value);
			其中使用previous_end_value.valid() 来检查是否有效，如果返回false就代表传入的是default_future*/

			block_start = block_last;
			block_start++;
			previous_end_values.push_back(end_values[i].get_future());
		}

		Iterator final_element = block_start;
		std::advance(final_element, std::distance(block_start, last) - 1);

		// 考虑num_threads > 1是为了特别考虑总共就需要一个thread的情况，此时就没有上一个块的累加。另外最后一个块是不需要promise的，
		process_chunk()(block_start, final_element, (num_threads > 1) ? &previous_end_values.back() :nullptr, nullptr);
		/*
		这里process_chunk()首先创建了一个临时默认构造对象，然后就是object(parameters)调用
		*/
	}

	// type without template
	//typedef std::vector<int>::iterator Iterator;
	//void parallel_partial_sum(Iterator first, Iterator last) {
	//	typedef Iterator::value_type value_type;
	//	struct process_chunk {
	//		void operator()(Iterator begin, Iterator last, std::future<value_type>* previous_end_value, std::promise<value_type>* end_value){
	//			try {
	//				Iterator end = last;
	//				++end;
	//				std::partial_sum(begin, end, begin);
	//				if (previous_end_value != nullptr) {
	//					//this is not the first thread
	//					auto addend = previous_end_value->get();
	//					*last += addend;
	//					if (end_value) {
	//						//not the last block
	//						end_value->set_value(*last);
	//					}
	//					std::for_each(begin, last, [addend](value_type& item) {
	//						item += addend;
	//						});
	//				}
	//				else if (end_value) {
	//					//this is the first thread
	//					end_value->set_value(*last);
	//				}
	//			}
	//			catch(...)
	//			{
	//				if (end_value) {
	//					//promise is available，即为不是最后一个block
	//					end_value->set_exception(std::current_exception());

	//				}
	//				else {
	//					//final block -main thread is the one process the final block
	//					throw;
	//				}

	//			}

	//		}
	//	};

	//	unsigned long const length = std::distance(first, last);
	//	if (!length)
	//		return;

	//	// 计算最优线程数
	//	unsigned long const min_per_thread = 25;
	//	unsigned long const max_threads = (length + min_per_thread - 1) / min_per_thread;
	//	unsigned long const hardware_threads = std::thread::hardware_concurrency();
	//	unsigned long const num_threads = std::min(hardware_threads != 0 ? hardware_threads : max_threads, max_threads);
	//	unsigned long const block_size = length / num_threads;

	//	//会用到的向量
	//	std::vector<std::thread> threads(num_threads - 1);
	//	std::vector<std::promise<Iterator::value_type>> end_values(num_threads - 1);	//向下一个thread发送累加值 ,这里展示了没有typedef应该怎么写
	//	std::vector<std::future<value_type>> previous_end_values;			//接收上一个thread的累加值
	//	previous_end_values.reserve(num_threads - 1);

	//	//主操作
	//	join_threads joiner(threads);

	//	Iterator block_start = first;
	//	for (unsigned long i = 0; i < (num_threads - 1); i++)
	//	{
	//		Iterator block_last = block_start;
	//		std::advance(block_last, block_size - 1);

	//		threads[i] = std::thread(process_chunk(), block_start, block_last, (i != 0) ? &previous_end_values[i - 1] : nullptr, &end_values[i]);

	//		block_start = block_last;
	//		block_start++;
	//		previous_end_values.push_back(end_values[i].get_future());
	//	}

	//	Iterator final_element = block_start;
	//	std::advance(final_element, std::distance(block_start, last) - 1);

	//	// 考虑num_threads > 1是为了特别考虑总共就需要一个thread的情况，此时就没有上一个块的累加。另外最后一个块是不需要promise的，
	//	process_chunk()(block_start, final_element, (num_threads > 1) ? &previous_end_values.back() :nullptr, nullptr);
	//}




	void run() {
		size_t Array_size = 1000;

		/*
											1	2	3	4	5	6			
		Inclusive scan						1	3	6	10	15	21
		Exclusive scan						0	1	3	6	10	15
		std::partial_sum(with add operator) 1	3	6	10	15	21

		*/

		std::vector<int> ints(Array_size);
		std::vector<int> outs(Array_size);
		for (auto& i : ints) {
			i = 1;
		}


		// 自己的线性实现
		print_function_run_time([&]() {
			std::cout << "/**********************************************/\n" << "自己实现的\n";
			sequential_partial_sum (ints.begin(), ints.end(), outs.begin());
		});


		print_function_run_time([&]() {
			std::cout << "/**********************************************/\n" << "inclusive_scan_sequential\n";
			std::inclusive_scan(ints.cbegin(), ints.cend(), outs.begin());
			});


		print_function_run_time([&]() {
			std::cout << "/**********************************************/\n" << "inclusive_scan_parallel\n";
			std::inclusive_scan(std::execution::par, ints.cbegin(), ints.cend(), outs.begin());
			});

		print_function_run_time([&]() {
			std::cout << "/**********************************************/\n" << "partial_sum_add_sequential,partial_sum只支持串行\n";
			std::partial_sum(ints.cbegin(), ints.cend(), outs.begin(),std::plus<int>());
			});

		//自己的并行实现,这里我们是传入副本，仅作计算没有返回结果
		print_function_run_time([&]() {
			std::cout << "/**********************************************/\n" << "my_parallel_partition\n";
			parallel_partial_sum(ints.begin(), ints.end());
			});


		
	}
}