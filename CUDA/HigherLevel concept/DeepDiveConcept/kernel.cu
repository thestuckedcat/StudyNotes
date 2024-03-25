
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
# include "copy_if.h"
#include <stdio.h>
# include<random>
# include<iostream>
# include<chrono>
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)


#define TIME_GPU(kernellauncher)	\
do						\
{						\
	cudaEvent_t start, stop;\
	cudaEventCreate(&start);\
	cudaEventCreate(&stop);\
	cudaEventRecord(start);\
	kernellauncher();\
	cudaEventRecord(stop);\
	cudaEventSynchronize(stop);\
	float milliseconds = 0;\
	cudaEventElapsedTime(&milliseconds,start, stop);\
	std::cout << "GPU Time taken: " << milliseconds / 1000 << " seconds" << std::endl;\
	cudaEventDestroy(start);\
	cudaEventDestroy(stop);\
} while (0)

#define TIME_CPU(func_call)                             \
do {                                                    \
    auto start = std::chrono::high_resolution_clock::now(); \
    func_call;                                          \
    auto stop = std::chrono::high_resolution_clock::now();  \
    std::chrono::duration<double> duration = stop - start;  \
    std::cout << "CPU Time taken: " << duration.count() << " seconds" << std::endl; \
} while (0)


void check_result(int* device_num, int* device_result, int* cpu_num, int* cpu_result) {
	if (*device_num != *cpu_num) {
		std::cout << "Device result is something wrong with cpu_result" << std::endl;
		std::cout << "device_num = " << *device_num << std::endl;
		std::cout << "cpu_num = " << *cpu_num << std::endl;

	}
	else {
		std::cout << "Device result is as same as cpu_result" << std::endl;
		/*
		int num = 0;
		for (int i = 0; i < *device_num; i++) {
			if (device_result[i] != cpu_result[i]) {
				num++;
				std::cout << "i-th result is not same where GPU is " << device_result[i] << " and CPU is " << cpu_result[i] << std::endl;
			}
		}
		std::cout << "The total no-match result num is " << num << std::endl;
		*/
	}
}

void compare1() {
	/*
	CPU vs original GPU copy_if
	*/
	const int array_size = 8192000;

	int* src = new int[array_size];


	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distrib(-100, 100);
	for (int i = 0; i < array_size; i++) {
		src[i] = distrib(gen);
		//std::cout << src[i] << " ";
	}

	int* res = new int[array_size];

	int resnum;
	TIME_CPU(resnum = cpu_copy_if(res, src, array_size));


	int* host_result1 = (int*)malloc(array_size * sizeof(int));
	int* host_nres1 = (int*)malloc(sizeof(int));
	int* device_src;
	int* device_result1;
	int* nres1;
	CHECK(cudaMalloc((void**)&device_src, array_size * sizeof(int)));
	CHECK(cudaMalloc((void**)&device_result1, array_size * sizeof(int)));
	CHECK(cudaMalloc((void**)&nres1, sizeof(int)));

	CHECK(cudaMemcpy(device_src, src, array_size * sizeof(int), cudaMemcpyHostToDevice));
	
	auto kernel_launcher = [&](){
		origin_copy_if << < 128, 512 >> >(device_result1, device_src, nres1, array_size);
	};

	TIME_GPU(kernel_launcher);

	CHECK(cudaMemcpy(host_result1, device_result1, sizeof(int) * array_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(host_nres1, nres1, sizeof(int), cudaMemcpyDeviceToHost));


	check_result(host_nres1, host_result1, &resnum, res);

	cudaFree(device_src);
	cudaFree(device_result1);
	cudaFree(nres1);
	free(host_result1);
	free(host_nres1);
	free(res);
	free(src);
}

void compare2() {
	/*
	CPU vs original GPU copy_if
	*/
	const int array_size = 8192000;

	int* src = new int[array_size];


	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distrib(-100, 100);
	for (int i = 0; i < array_size; i++) {
		src[i] = distrib(gen);
		//std::cout << src[i] << " ";
	}

	int* res = new int[array_size];

	int resnum;
	TIME_CPU(resnum = cpu_copy_if(res, src, array_size));


	int* host_result1 = (int*)malloc(array_size * sizeof(int));
	int* host_nres1 = (int*)malloc(sizeof(int));
	int* device_src;
	int* device_result1;
	int* nres1;
	CHECK(cudaMalloc((void**)&device_src, array_size * sizeof(int)));
	CHECK(cudaMalloc((void**)&device_result1, array_size * sizeof(int)));
	CHECK(cudaMalloc((void**)&nres1, sizeof(int)));

	CHECK(cudaMemcpy(device_src, src, array_size * sizeof(int), cudaMemcpyHostToDevice));

	auto kernel_launcher = [&]() {
		blocklevel_copy_if << < 128,512 >> > (device_result1, device_src, nres1, array_size);
		};

	TIME_GPU(kernel_launcher);

	CHECK(cudaMemcpy(host_result1, device_result1, sizeof(int) * array_size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(host_nres1, nres1, sizeof(int), cudaMemcpyDeviceToHost));


	check_result(host_nres1, host_result1, &resnum, res);

	cudaFree(device_src);
	cudaFree(device_result1);
	cudaFree(nres1);
	free(host_result1);
	free(host_nres1);
	free(res);
	free(src);
}

int main()
{
	compare1();
	compare2();




}