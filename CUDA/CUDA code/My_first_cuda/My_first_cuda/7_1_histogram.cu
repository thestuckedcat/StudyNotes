# include<stdio.h>
# include<cuda.h>
# include"cuda_runtime.h"

template<int blockSize>
__global__ void histgram(int* hist_data, int* bin_data, int N) {
	__shared__ int cache[256];

	int gtid = blockIdx.x * blockSize + threadIdx.x;
	int tid = threadIdx.x;

	cache[tid] = 0;
	__syncthreads();

	for (int i = gtid; i < N; i += gridDim.x * blockSize) {
		atomic
	}

}