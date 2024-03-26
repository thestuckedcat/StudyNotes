# include "copy_if.h"
# include <iostream>
# include <cuda.h>
# include <cuda_runtime.h>
# include <limits.h>

__global__ void origin_copy_if(int* res, const int* src, int* nres, int n) {
	// ���ַ����õ���res������ѭԭsrc�з��������������е�˳��
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
	// block�ڵļ���
	__shared__ int count;

	int gridsize = gridDim.x * blockDim.x;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int pos;
	// ѭ������
	for (int i = idx; i < n; i+= gridsize) {
		//��ֹƵ������λ���Դ��src
		int data = src[i];

		// ��ÿ��block�У�ָ��һ���߳����ȳ�ʼ��count
		if (threadIdx.x == 0) {
			count = 0;
		}
		__syncthreads();

		// ÿ���߳��ж��Ƿ�����Ҫ��
		if (i < n && data > 0) {
			//����Ҫ���ʹ��shared_memory variable��¼,pos��¼��ƫ��
			pos = atomicAdd(&count, 1);
		}

		__syncthreads();

		// ȫ����ɺ���һ���߳������Լ�block����Ϣ������block���ϣ���ȡblock�����ƫ��
		// ����countʵ�����Ǻ����ò����ˣ�ֱ�Ӹ�����Ϊbias
		// block��˳����ȫȡ�������block��thread0�ܹ����ϵ�ʱ��
		if (threadIdx.x == 0) {
			count = atomicAdd(nres, count);
		}
		__syncthreads();
		//��ʱ�����߳̿��Ի�ȡ����ȫ�ֵ�Ψһ����
		if (i < n && data > 0) {
			res[count + pos] = data;
		}

		__syncthreads();



	}

}


//warp level copy if

__device__ int atomic_allocate_index(int* nres) {
	//��ȡ��ǰwarp��32λ�Ļ�Ծ�߳�mask�����ȫ��Ծ����0xFFFFFFFF
	unsigned int active_thread_mask = __activemask();

	// ��ȡ��Ծ�߳�������Ҳ���Ƿ���if����������)
	int active_thread_num = __popc(active_thread_mask);

	// ʹ�û����lanemask_lt�����ֵ��32λ��ʾ��
	// ������ʾΪ�������warp�ڵ�����̣߳���ô���ǵ���λ�߳���Ϊ1������ȫΪ0��00...0001111
	// Ҳ����˵�������ܹ�ֱ��ʹ��__popc(lanemask_lt)�õ�����̵߳�ID(0-31)
	unsigned int lane_mask_in_warp;
	asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_in_warp));

	// ��ȡ����if���߳�����, Ҳ������Ҫ��һλͬʱ�ǻ�Ծ�߳�(��1)�Լ��ڴ��߳�֮ǰ
	// Ҳ����ͳ�ƻ�Ծ�̵߳�ƫ��
	int inwarp_offset = __popc(active_thread_mask & lane_mask_in_warp);

	int global_offset;

	// ѡ���߳�0Ϊleader�߳�
	if (inwarp_offset == 0)
	{
		// ���ȫ��ƫ��
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