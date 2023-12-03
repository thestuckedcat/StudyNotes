#pragma once
# include<stddef.h>
# include <stdio.h>
# include <algorithm>
# include <random>
# include <ratio>
# include <vector>
# include <chrono>
# include <execution>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::milli;
using std::random_device;
using std::sort;
using std::vector;
namespace using_parallel_stl {
	const size_t testSize = 1000000;
	const int iterationCount = 5;

	void print_results(const char* const tag, const vector<double>& sorted, high_resolution_clock::time_point startTIme, high_resolution_clock::time_point endTime) {
		printf("%s : Lowest : %lf Highest: %lf Time: %lld\n", tag, sorted.front(), sorted.back(), duration_cast<std::chrono::milliseconds>(endTime - startTIme));
	}

	void run() {
		random_device rd;

		// generate some random doubles
		printf("Testing with %zu doubles...\n", testSize);
		vector<double> doubles(testSize);
		for (auto& d : doubles) {
			d = static_cast<double>(rd());
		}

		//time how long it takes to sort them,为了数据稳定，我们应该选取多次运行的average run time
		for (int i = 0; i < iterationCount; ++i) {
			vector<double> sorted(doubles);
			const auto startTime = high_resolution_clock::now();
			sort(sorted.begin(), sorted.end());
			const auto endTime = high_resolution_clock::now();
			print_results("Serial STL ", sorted, startTime, endTime);
		}

		for (int i = 0; i < iterationCount; i++) {
			vector<double> sorted(doubles);
			const auto startTime = high_resolution_clock::now();
			std::sort(std::execution::par, sorted.begin(), sorted.end());
			const auto endTime = high_resolution_clock::now();
			print_results("Parallel STL", sorted, startTime, endTime);
		}
	}
}