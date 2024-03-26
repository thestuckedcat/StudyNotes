# include "copy_if.h"
# include <iostream>
# include <cuda.h>
# include <cuda_runtime.h>
# include <limits.h>

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


//warp level copy if

__device__ int atomic_allocate_index(int* nres) {
	//获取当前warp内32位的活跃线程mask，如果全活跃就是0xFFFFFFFF
	unsigned int active_thread_mask = __activemask();

	// 获取活跃线程数量（也就是符合if条件的数量)
	int active_thread_num = __popc(active_thread_mask);

	// 使用汇编获得lanemask_lt，这个值用32位表示，
	// 其具体表示为，如果是warp内第五个线程，那么就是低四位线程置为1，其余全为0，00...0001111
	// 也就是说，我们能够直接使用__popc(lanemask_lt)得到这个线程的ID(0-31)
	unsigned int lane_mask_in_warp;
	asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_in_warp));

	// 获取符合if的线程数量, 也就是需要这一位同时是活跃线程(置1)以及在此线程之前
	// 也就是统计活跃线程的偏置
	int inwarp_offset = __popc(active_thread_mask & lane_mask_in_warp);

	int global_offset;

	// 选择线程0为leader线程
	if (inwarp_offset == 0)
	{
		// 获得全局偏置
		global_offset = atomicAdd(nres, active_thread_num);
	}

	//__syncwarp();
	global_offset = __shfl_sync(active_thread_mask, global_offset, 0);

	return global_offset + inwarp_offset;

}

__global__ void warplevel_copy_if(int* res, const int* src, int* nres, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int gridsize = gridDim.x * blockDim.x;

	for (int i = idx; i < n; i += gridsize) {
		if (src[i] > 0) {
			res[atomic_allocate_index(nres)] = src[i];
		}
	}
}