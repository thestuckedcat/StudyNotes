#pragma once
# include<iostream>
# include<vector>
# include<numeric>
# include<string>
# include<algorithm>
# define MIN_BLOCK_SIZE 1000
namespace parallel_accumulation {

	void sequntial_accumulation_test() 
	{
		//չʾ��ʹ��std::accumulation�㷨��Iterator��������ۼ�
		std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
		int sum = std::accumulate(v.begin(), v.end(), 0);

		std::cout << "Using default std::accumulate+ sum=" << sum << std::endl;

		int product = std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());

		std::cout << "Using self_define Binary operation* sum=" << product << std::endl;

		auto dash_fold = [](std::string a, int b) {
			return std::move(a) + "-" + std::to_string(b);
		};
		std::string s = std::accumulate(std::next(v.begin()), v.end(), std::to_string(v[0]), dash_fold);
		std::cout << "Using lambda_expression Binary operation to add - between number: " << s << std::endl;

	}


	template<typename iterator, typename T>
	void inthread_accumulate(iterator start, iterator end, T& ref) 
	{
		ref = std::accumulate(start, end, 0);
	}


	template<typename iterator, typename T>
	T parallel_accumulate(iterator start, iterator end, T &ref) 
	{
		int input_size = std::distance(start, end);//end-start:ֻ��������ʵ�����������-������˫���������ǰ�������������֧��ֱ�ӵ���������
		int allowed_threads_by_elements = (input_size) / MIN_BLOCK_SIZE;

		int allowed_threads_by_hardware = std::thread::hardware_concurrency();//��Ҫ����Ӳ��ˮƽ

		int num_threads = std::min(allowed_threads_by_elements, allowed_threads_by_hardware);
		std::cout << num_threads << std::endl;
		int block_size = (input_size+num_threads-1) / num_threads;


		std::vector<T> results(num_threads);
		std::vector<std::thread> threads(num_threads-1);

		iterator last;
		for (int i = 0; i < num_threads - 1; i++) {
			last = start;
			std::advance(last, block_size);//�൱��ָ���ptr+blocksize��ֻ������������Ҫadvance
			threads[i] = std::thread(inthread_accumulate<iterator, T>, start, last, std::ref(results[i]));
			start = last;//accumulate������start������end��һ��iterator.end()ָ��������һ��Ԫ�صĺ�һλ
		}

		results[num_threads - 1] = std::accumulate(start, end, 0);//����ʣ�������

		
		std::for_each(threads.begin(), threads.end(), [](std::thread& t) {
			if (t.joinable()) {
				t.join();
			}
		});

		return std::accumulate(results.begin(), results.end(),ref);
	}

	void run_parallel_accumulate() {
		const int size = 7998;
		int* my_array = new int[size];
		int ref = 0;

		srand(0);

		for (size_t i = 0; i < size; i++) {
			my_array[i] = 1;
			//my_array[i] = rand() % 10;
		}

		int rer_val = parallel_accumulate<int*, int>(my_array, my_array + size, ref);
		printf("Accumulated value : %d \n", rer_val);
	}
	void run() {
		//sequntial_accumulation_test();

		run_parallel_accumulate();
		
	}
}