#pragma once
# include <iostream>
# include <thread>
# include <algorithm>
# include <vector>
# include <memory>
# include <stdexcept>
# include <chrono>
# include <random>
namespace parallel_matrix {
	
	class thread_joiner {
		std::vector<std::thread>& threads;

	public:
		explicit thread_joiner(std::vector<std::thread>& _threads) :threads{ _threads } {}

		~thread_joiner() {
			for (auto& t : threads) {
				if (t.joinable()) {
					t.join();
				}
			}
		}
	};

	class Matrix {

		int* data;
		int rows;		// row count	length of columns
		int columns;	// column count length of rows

	public:

		Matrix(int _n, int _m) : rows(_n), columns(_m)
		{
			data = new int[rows * columns];
			//set the array to 0
			std::fill(data, data + rows * columns, 0);
		}
		Matrix(int _n, int _m, int* _data) : rows(_n), columns(_m), data(_data)
		{
			_data = nullptr;
		}

		// i -> 0 to n-1
		// j -> 0 to m-1
		void set_value(int i, int j, int value)
		{
			data[i * columns + j] = value;
		}

		void set_all(int value)
		{
			std::fill(data, data + rows * columns, value);
		}

		static void multiply(Matrix* x, Matrix* y, Matrix* results)
		{
			//check the matrix sizes are correct to multiply
			if (!(x->columns == y->rows) || !((x->rows == results->rows) && (y->columns == results->columns)))
			{
				std::cout << " ERROR : Invalid matrix sizes for multiplication \n";
				return;
			}

			// r = result_size
			int r = results->rows * results->columns;

			for (size_t i = 0; i < r; i++)
			{
				for (size_t j = 0; j < x->columns; j++)
				{
					results->data[i] += x->data[(i / results->columns) * x->columns + j]
						* y->data[i % results->columns + j * y->columns];
				}
			}
		}

		static void parallel_multiply(Matrix* x, Matrix* y, Matrix* results) {
			// thread call function
			struct process_data_chunk {
				void operator()(Matrix* x, Matrix* y, Matrix* results, int begin, int end) {
					for (size_t i = begin; i < end; i++) {
						for (size_t j = 0; j < x->columns; j++) {
							results->data[i] += x->data[(i / results->columns) * results->columns + j] * y->data[i % results->columns + j * y->columns];
						}
					}
				}
			};
			// check correction
			if (!((x->rows == results->rows) && (y->columns == results->columns)) || !(x->columns == y->rows))
			{
				std::cout << " ERROR : Invalid matrix sizes for multiplication \n";
			}

			//calculate result size
			int length = results->rows * results->columns;

			if (!length)
				return;

			// calculate optimal thread num
			const int min_data_per_thread = 10000;
			int max_threads = (length + min_data_per_thread - 1) / min_data_per_thread;;
			int hardware_thread_num = std::thread::hardware_concurrency();
			int thread_num = std::min(hardware_thread_num == 0 ? 1 : hardware_thread_num, max_threads);
			int block_size = length / thread_num;
			//std::cout << "总线程数量为" << length << " 分配线程数量为" << (block_size * thread_num) << "最后一个线程数量为 " << (length - block_size * (thread_num-1)) << std::endl;;
			//vectorize thread
			std::vector<std::thread> threads(thread_num-1);

			//use a new block allocate threads
			{
				int block_start = 0;

				thread_joiner thread_join(threads);

				for (int i = 0; i < thread_num - 1; i++) {
					threads[i] = std::thread(process_data_chunk(), x,y,results,block_start, block_start + block_size);//这里不需要减一，因为遍历不包括end，bug找半天md
					block_start += block_size;

				}

				// process the remaining data
				process_data_chunk()(x, y, results, block_start, length);

			}
		}

		static void transpose(Matrix* x, Matrix* result) {
			// check correction
			if (x->rows != result->columns || x->columns != result->rows) {
				std::cout << "There is something wrong with result size\n";
				return;
			}

			int length = result->rows * result->columns;

			for (size_t i = 0; i < length; i++) {
				int result_row_index = i / result->columns;
				int result_column_index = i % result->rows;

				int input_row_index = result_column_index;
				int input_column_index = result_row_index;

				result->data[i] = x->data[input_row_index * x->columns + input_column_index];
			}
		}

		static void parallel_transpose(Matrix* x, Matrix* result) 
		{
			// thread call function
			struct thread_process_call {
				void operator()(Matrix* A, Matrix* result, int begin, int end) {
					for (size_t i = begin; i < end; i++) {
						int input_column_index = i / result->columns;
						int input_row_index = i % result->columns;

						result->data[i] = A->data[input_row_index * A->columns + input_column_index];
					}
				}
			};
			// check correction
			if (x->columns != result->rows || x->rows != result->columns) {
				std::cout << " result的size不对\n";
				return;
			}

			//calculate result size
			int length = result->columns * result->rows;
			if (!length) {
				return;
			}

			// calculate optimal thread num
			int min_data_per_thread = 10000;
			int max_thread_num = (length + min_data_per_thread - 1) / min_data_per_thread;
			int hardware_thread_num = std::thread::hardware_concurrency();
			int thread_num = std::min(hardware_thread_num == 0 ? 2 : hardware_thread_num, max_thread_num);
			int block_size = length / thread_num;


			
			//vectorize thread
			std::vector<std::thread> threads(thread_num - 1);


			//use a new block allocate threads
			{
				thread_joiner joiner(threads);
				int begin = 0;
				for (int i = 0; i < thread_num - 1; i++) {
					threads[i] = std::thread(thread_process_call(), x, result, begin, begin + block_size);
					begin += block_size;
				}
				thread_process_call()(x, result, begin, length);
			}
		}


		/*
		static void parallel_multiply(Matrix* x, Matrix* y, Matrix* results)
		{
			struct process_data_chunk
			{
				void operator()(Matrix* results, Matrix* x, Matrix* y, int start_index, int end_index)
				{
					for (size_t i = start_index; i < end_index; i++)
					{
						for (size_t j = 0; j < x->columns; j++)
						{
							results->data[i] += x->data[(i / results->columns) * x->columns + j]
								* y->data[i % results->rows + j * y->columns];
						}
					}
				}

			};

			//check the matrix sizes are correct to multiply
			if (!((x->rows == results->rows) && (y->columns == results->columns)) || !(x->columns == y->rows))
			{
				std::cout << " ERROR : Invalid matrix sizes for multiplication \n";
			}

			// r = result_size
			int length = results->rows * results->columns;

			if (!length)
				return;

			int min_per_thread = 10000;
			int max_threads = (length + min_per_thread - 1) / min_per_thread;
			int hardware_threads = std::thread::hardware_concurrency();
			int num_threads = std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);
			int block_size = length / num_threads;

			std::vector<std::thread> threads(num_threads - 1);
			int block_start = 0;
			int block_end = 0;
			{
				join_threads joiner(threads);

				for (unsigned long i = 0; i < (num_threads - 1); i++)
				{
					block_start = i * block_size;
					block_end = block_start + block_size;
					threads[i] = std::thread(process_data_chunk(), results, x, y, block_start, block_end);
				}

				// perform the find operation for final block in this thread.
				process_data_chunk()(results, x, y, block_end, length);
			}
		}

		static void transpose(Matrix* x, Matrix* results)
		{
			//check the matrix sizes are correct to multiply
			if (!((x->columns == results->rows) && (x->rows == results->columns)))
			{
				std::cout << " ERROR : Invalid matrix sizes for transpose \n";
				return;
			}

			// r = result_size
			int r = results->rows * results->columns;

			int result_column = 0;
			int result_row = 0;

			int input_column = 0;
			int input_row = 0;

			for (size_t i = 0; i < r; i++)
			{
				//get the current row and column count
				result_row = i / results->columns;
				result_column = i % results->columns;

				//flipped the columns and row for input
				input_row = result_column;
				input_column = result_row;

				//store the corresponding element from input to the results
				results->data[i] = x->data[input_row * x->columns + input_column];
			}
		}

		static void parallel_transpose(Matrix* x, Matrix* results)
		{
			struct process_data_chunk
			{
				void operator()(Matrix* results, Matrix* x, int start_index, int end_index)
				{
					int result_column = 0;
					int result_row = 0;

					int input_column = 0;
					int input_row = 0;

					for (size_t i = start_index; i < end_index; i++)
					{
						result_row = i / results->columns;
						result_column = i % results->columns;

						input_row = result_column;
						input_column = result_row;

						results->data[i] = x->data[input_row * x->columns + input_column];
					}
				}

			};

			//check the matrix sizes are correct to multiply
			if (!((x->columns == results->rows) && (x->rows == results->columns)))
			{
				std::cout << " ERROR : Invalid matrix sizes for transpose \n";
				return;
			}

			// r = result_size
			int length = results->rows * results->columns;

			if (!length)
				return;

			int min_per_thread = 10000;
			int max_threads = (length + min_per_thread - 1) / min_per_thread;
			int hardware_threads = std::thread::hardware_concurrency();
			int num_threads = std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);
			int block_size = length / num_threads;

			std::vector<std::thread> threads(num_threads - 1);
			int block_start = 0;
			int block_end = 0;
			{
				join_threads joiner(threads);

				for (unsigned long i = 0; i < (num_threads - 1); i++)
				{
					block_start = i * block_size;
					block_end = block_start + block_size;
					threads[i] = std::thread(process_data_chunk(), results, x, block_start, block_end);
				}

				// perform the find operation for final block in this thread.
				process_data_chunk()(results, x, block_end, length);
			}
		}
		*/
		void print()
		{
			if (rows < 50 && columns < 50)
			{
				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < columns; j++)
					{
						std::cout << data[j + i * columns] << " ";
					}

					std::cout << "\n";
				}
				std::cout << std::endl;
			}
		}

		int query_rows() const {
			return this->rows;
		}
		int query_columns() const {
			return this->columns;
		}
		bool operator==(const Matrix &rhs) const {
			if (this->rows == rhs.query_rows() && this->columns == rhs.query_columns()) {
				for (size_t i = 0; i < rows * columns; i++)
				{
					//std::cout << data[i] << " " << rhs.data[i] << std::endl;
					if (data[i] != rhs.data[i])
					{
						std::cout << "element error\n";
						return false;
					}
				}
				return true;
			}
			std::cout << "size error\n";
			return false;
		}
		static bool is_transpose(const Matrix* x, const Matrix* result) {
			if (x->rows != result->columns || x->columns != result->rows) {
				std::cout << "There is something wrong with result size\n";
				return false;
			}
			for (int i = 0; i < x->rows; i++) {
				for (int j = 0; j < x->columns; j++) {
					if (x->data[i * x->columns + j] != result->data[j * result->columns + i]) {
						std::cout << " element error\n";
						return false;
					}
				}
			}
			return true;

		}



		~Matrix()
		{
			delete data;
		}
	};

	template<typename func>
	void print_time(const char* tag, func callable_func) {
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		callable_func();
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		printf("%s, Time %lf ms\n", tag, std::chrono::duration_cast<std::chrono::duration<double, std::milli>> (end - start).count());
	}

	
	void run_multiple() {
		const int matrix_size = 2000;
		int* dataA = new int[matrix_size * matrix_size];
		int* dataB = new int[matrix_size * matrix_size];

		std::random_device rd;		//生成一个与设备相关的种子
		std::mt19937 gen(rd());		//随机数生成器利用种子生成随机数
		std::uniform_int_distribution<> dis(1, 100);	//定义一个分布
		for (int i = 0; i < matrix_size * matrix_size; i++) {
			dataA[i] = dis(gen);//生成一个随机数
			dataB[i] = dis(gen);
		}
		Matrix A(matrix_size, matrix_size,dataA);
		Matrix B(matrix_size, matrix_size,dataB);
		Matrix C(matrix_size, matrix_size);
		Matrix D(matrix_size, matrix_size);


		print_time(" Sequential Multiply ", [&]() {
			Matrix::multiply(&A, &B, &C);
			});

		print_time(" Parallel Multiply", [&]() {
			Matrix::parallel_multiply(&A, &B, &D);
			});

		std::cout << (C == D) << std::endl;
		
	}

	void run_transpose() {
		const int matrix_size = 20000;
		int* dataA = new int[matrix_size * matrix_size];

		std::random_device rd;		//生成一个与设备相关的种子
		std::mt19937 gen(rd());		//随机数生成器利用种子生成随机数
		std::uniform_int_distribution<> dis(1, 100);	//定义一个分布
		for (int i = 0; i < matrix_size * matrix_size; i++) {
			dataA[i] = dis(gen);//生成一个随机数
		}
		Matrix A(matrix_size, matrix_size, dataA);
		Matrix C(matrix_size, matrix_size);
		Matrix D(matrix_size, matrix_size);


		print_time(" Sequential transpose ", [&]() {
			Matrix::transpose(&A,&C);
			});

		print_time("parallel transpose ", [&]() {
			Matrix::parallel_transpose(&A, &D);
			});


		std::cout << Matrix::is_transpose(&A, &C) << std::endl;
		std::cout << (C == D) << std::endl;
	}
}