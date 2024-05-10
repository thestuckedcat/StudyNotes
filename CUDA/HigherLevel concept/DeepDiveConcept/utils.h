#pragma once
#include<iostream>
#include<chrono>

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


