# include "copy_if.h"
# include <iostream>
# include <cuda.h>
# include <cuda_runtime.h>

__global__ void origin_copy_if(int* res, const int* src, int* nres, int n) {
	// 这种方法得到的res并不遵循原src中符合条件的数固有的顺序
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int total_size = gridDim.x * blockDim.x;
	int loop = n/total_size + 1;

	for(int i = 0; i < loop; i++)
	{
		int true_idx = idx + total_size * i;
		if ( true_idx < n && src[true_idx] > 0) {
			res[atomicAdd(nres, 1)] = src[true_idx];
		}
	}

}

__host__ int cpu_copy_if(int* res, const int* src, int n) {

	int resnum = 0;
	for (int i = 0; i < n; i++) {
		if (src[i] > 0) {
			res[resnum++] = src[i];
		}
	}
	return resnum;
}

__global__ void blocklevel_copy_if(int* res, const int* src, int* nres, int n) {
	// block内的计数
	__shared__ int count;

	int gridsize = gridDim.x * blockDim.x;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int pos;
	// 循环处理
	for (int i = idx; i < n; i+= gridsize) {
		//防止频繁访问位于显存的src
		int data = src[i];

		// 在每个block中，指定一个线程首先初始化count
		if (threadIdx.x == 0) {
			count = 0;
		}
		__syncthreads();

		// 每个线程判断是否满足要求
		if (i < n && data > 0) {
			//满足要求的使用shared_memory variable记录,pos记录了偏置
			pos = atomicAdd(&count, 1);
		}

		__syncthreads();

		// 全部完成后，让一个线程来将自己block的信息与其他block整合，获取block级别的偏置
		// 这里count实际上是后面用不到了，直接复用作为bias
		// block的顺序完全取决于这个block的thread0能够加上的时机
		if (threadIdx.x == 0) {
			count = atomicAdd(nres, count);
		}
		__syncthreads();
		//此时所有线程可以获取他们全局的唯一坐标
		if (i < n && data > 0) {
			res[count + pos] = data;
		}

		__syncthreads();



	}

}