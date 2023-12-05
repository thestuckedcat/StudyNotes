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
		typename������֮һ
		��ģ�������ָ���������ͣ�
		����ģ�����������һ��������ģ�����������ʱ����Ҫʹ�� typename��
		������Ϊ��ģ��ʵ����֮ǰ�������������޷�ȷ��ĳ�������Ƿ��ʾһ�����͡�
		���磬�� typename MyTemplate<T>::SubType x; �У�typename ���ڸ��߱����� MyTemplate<T>::SubType ��һ������
		�ڱ������У���Ϊ������Ҫʹ����������prmomise��future���䷵��ֵ��δ�����Iterator��أ����������Ҫʹ��typename Iterator::value_type��ָ��
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
						//promise is available����Ϊ�������һ��block
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

		// ���������߳���
		unsigned long const min_per_thread = 25;
		unsigned long const max_threads = (length + min_per_thread - 1) / min_per_thread;
		unsigned long const hardware_threads = std::thread::hardware_concurrency();
		unsigned long const num_threads = std::min(hardware_threads != 0 ? hardware_threads : max_threads, max_threads);
		unsigned long const block_size = length / num_threads;

		//���õ�������
		std::vector<std::thread> threads(num_threads - 1);
		std::vector<std::promise<typename Iterator::value_type>> end_values(num_threads - 1);	//����һ��thread�����ۼ�ֵ ,����չʾ��û��typedefӦ����ôд
		std::vector<std::future<value_type>> previous_end_values;			//������һ��thread���ۼ�ֵ
		previous_end_values.reserve(num_threads - 1);
		/*
		����ʹ��reserve����ֱ�ӹ��죬ԭ������

		1.reserve ������������κ�Ԫ�صĹ��캯����reserve �����ڵ���������������Ҳ���������Դ洢��Ԫ�������������ı�������ʵ�ʴ�С����������Ԫ�ص���������
		�������� reserve ʱ��std::vector ������㹻���ڴ��Դ洢ָ��������Ԫ�أ������ᴴ�����ʼ����ЩԪ�ء�
		����ζ�� reserve �������� size() ��Ȼ���ֲ��䣬ֻ���� capacity() �����ˡ�

		������������ֱ��ʹ��previous_end_values(num_threads-1),���Ǻ����Ӧ��ʹ��previous_end_values[flag++] = end_values[i].get_future();�˴����ƶ�����
		�������汾�����ڵİ汾����������ռ��һ��push
		ֱ�ӹ���汾����������ռ䣬��ʼ�����쿪���Լ��ƶ���ֵ���죬��Щ��������������ʱ���ݺ���
		��ˣ�������Ԫ������ӵĳ����У�ʹ��reserve���Ӻ���

		2. �Ա���һ��resize

		��ͬ�� reserve��resize ������ı� std::vector ��ʵ�ʴ�С����������Ԫ�ص�������
		������� resize ʱ������µĴ�С���ڵ�ǰ��С��std::vector ������㹻������Ԫ���Դﵽָ���Ĵ�С��
		��Щ����ӵ�Ԫ�ػᱻĬ�Ϲ��죬������� resize �ṩ��һ��ֵ��Ϊ�ڶ�����������Ԫ�ؽ�����ʼ��Ϊ��ֵ��
		
		*/

		//������
		join_threads joiner(threads);

		Iterator block_start = first;
		for (unsigned long i = 0; i < (num_threads - 1); i++)
		{
			Iterator block_last = block_start;
			std::advance(block_last, block_size - 1);

			threads[i] = std::thread(process_chunk(), block_start, block_last, (i != 0) ? &previous_end_values[i - 1] : nullptr, &end_values[i]);

			/*�����еڶ���д��,��ȡ���ô��ݣ������鷳����Ҳ����������ʹ��ָ�봫�ݵķ���֮��
			std::future<value_type> default_future;
			threads[i] = std::thread(process_chunk(), block_start, block_last, (i != 0) ? std::ref(previous_end_values[i - 1]) : default_future, std::ref(end_values[i]));
			��Ӧ�ĺ���
			void operator()(Iterator begin, Iterator last, std::future<value_type> &previous_end_value, std::promise<value_type> &end_value);
			����ʹ��previous_end_value.valid() ������Ƿ���Ч���������false�ʹ��������default_future*/

			block_start = block_last;
			block_start++;
			previous_end_values.push_back(end_values[i].get_future());
		}

		Iterator final_element = block_start;
		std::advance(final_element, std::distance(block_start, last) - 1);

		// ����num_threads > 1��Ϊ���ر����ܹ�����Ҫһ��thread���������ʱ��û����һ������ۼӡ��������һ�����ǲ���Ҫpromise�ģ�
		process_chunk()(block_start, final_element, (num_threads > 1) ? &previous_end_values.back() :nullptr, nullptr);
		/*
		����process_chunk()���ȴ�����һ����ʱĬ�Ϲ������Ȼ�����object(parameters)����
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
	//					//promise is available����Ϊ�������һ��block
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

	//	// ���������߳���
	//	unsigned long const min_per_thread = 25;
	//	unsigned long const max_threads = (length + min_per_thread - 1) / min_per_thread;
	//	unsigned long const hardware_threads = std::thread::hardware_concurrency();
	//	unsigned long const num_threads = std::min(hardware_threads != 0 ? hardware_threads : max_threads, max_threads);
	//	unsigned long const block_size = length / num_threads;

	//	//���õ�������
	//	std::vector<std::thread> threads(num_threads - 1);
	//	std::vector<std::promise<Iterator::value_type>> end_values(num_threads - 1);	//����һ��thread�����ۼ�ֵ ,����չʾ��û��typedefӦ����ôд
	//	std::vector<std::future<value_type>> previous_end_values;			//������һ��thread���ۼ�ֵ
	//	previous_end_values.reserve(num_threads - 1);

	//	//������
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

	//	// ����num_threads > 1��Ϊ���ر����ܹ�����Ҫһ��thread���������ʱ��û����һ������ۼӡ��������һ�����ǲ���Ҫpromise�ģ�
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


		// �Լ�������ʵ��
		print_function_run_time([&]() {
			std::cout << "/**********************************************/\n" << "�Լ�ʵ�ֵ�\n";
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
			std::cout << "/**********************************************/\n" << "partial_sum_add_sequential,partial_sumֻ֧�ִ���\n";
			std::partial_sum(ints.cbegin(), ints.cend(), outs.begin(),std::plus<int>());
			});

		//�Լ��Ĳ���ʵ��,���������Ǵ��븱������������û�з��ؽ��
		print_function_run_time([&]() {
			std::cout << "/**********************************************/\n" << "my_parallel_partition\n";
			parallel_partial_sum(ints.begin(), ints.end());
			});


		
	}
}