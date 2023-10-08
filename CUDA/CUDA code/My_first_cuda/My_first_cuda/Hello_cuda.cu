# include "cuda_runtime.h"
# include <cuda.h>
# include "device_launch_parameters.h"
# include "common.h"
# include <cmath>
# include <iostream>

# include<stdio.h>
# include<stdlib.h>
# include<time.h>

# include<cstring>
using namespace std;
# define get_runtime(device, runtime, block_size) {printf("Algorithm %s execution time : %4.6f in block_size = %d \n", device, runtime, block_size); }
# define gpuErrchk(ans) {gpuAssert((ans), __FILE__,__LINE__);}

// add a kernel
__global__ void hello_cuda()
{
	printf("Hello CUDA world \n");
}
// a new kernel shows 唯一标识符 of thread
__global__ void print_threadIds()
{
	printf("threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d\n",
		threadIdx.x, threadIdx.y, threadIdx.z);
}
__global__ void print_key_words_detail() {
	printf("threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d, blockIdx.x : %d, blockIdx.y : %d, blockIdx.z : %d, blockDim.x : %d, blockDim.y : %d, blockDim.z : %d, gridDim.x : %d, gridDim.y : %d, gridDim.z : %d\n",
		threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}
// access data from kernel
__global__ void unique_idx_calc_threadIdx(int *input)
{
	printf("threadIdx : %d, value : %d \n", threadIdx.x, input[threadIdx.x]);
}
__global__ void unique_gid_calc_threadIdx(int* input) {
	int global_index = threadIdx.x + blockIdx.x * blockDim.x;
	printf("blockIdx.x : %d, threadIdx.x : %d, gid : %d, value : %d \n",
		blockIdx.x, threadIdx.x, global_index, input[global_index]);
}

// memory transfer
__global__ void memory_transfer_test(int* input)
{
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	printf("tid:%d, gid : %d, value : %d \n", threadIdx.x, global_index, input[global_index]);
}

// array summury: sum two (n*1*1) array
__global__ void sum_array_gpu(int* a, int* b, int* c, int size) 
{
	size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_index < size) {
		c[global_index] = a[global_index] + b[global_index];
	}
}
void sum_array_cpu(int* a, int* b, int* c, int size)
{
	//this is a comparitive function
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}

//homework 2
__global__ void sum_three_array_gpu(int * fir, int * sec, int * thir, int * ans, int size) {
	size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_index < size) {
		ans[global_index] = fir[global_index] + sec[global_index] + thir[global_index];
	}
}

void sum_three_array_cpu(int * fir, int * sec, int * thir, int * ans, int size) {
	for (size_t i{ 0 }; i < size; i++)
	{
		ans[i] = fir[i] + sec[i] + thir[i];
	}
}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert : %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void execute_homework2(int block_size) {
	// homework2
	clock_t cpu_start, cpu_end, gpu_start, gpu_end;

	int size = 1 << 22;
	cudaError error;
	int NO_BYTES = (int)sizeof(int) * size;

	// host pointer
	int* h_fir, * h_sec, * h_thir, * gpu_result, * cpu_result;
	h_fir = (int*)malloc(NO_BYTES);
	h_sec = (int*)malloc(NO_BYTES);
	h_thir = (int*)malloc(NO_BYTES);
	gpu_result = (int*)malloc(NO_BYTES);
	cpu_result = (int*)malloc(NO_BYTES);

	// initialize host pointer
	time_t t;
	srand((unsigned int)time(&t));
	for (size_t i{ 0 }; i < size; i++)
	{
		h_fir[i] = (int)(rand() & 0xFF);
	}
	for (size_t i{ 0 }; i < size; i++)
	{
		h_sec[i] = (int)(rand() & 0xFF);
	}
	for (size_t i{ 0 }; i < size; i++)
	{
		h_thir[i] = (int)(rand() & 0xFF);
	}
	cpu_start = clock();
	sum_three_array_cpu(h_fir, h_sec, h_thir, cpu_result, size);
	cpu_end = clock();
	get_runtime("CPU", (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC), block_size);
	memset(gpu_result, 0, NO_BYTES);

	// device pointer
	int* d_fir, * d_sec, * d_thir, * d_result;
	gpuErrchk(cudaMalloc((int**)&d_fir, NO_BYTES));
	gpuErrchk(cudaMalloc((int**)&d_sec, NO_BYTES));
	gpuErrchk(cudaMalloc((int**)&d_thir, NO_BYTES));
	gpuErrchk(cudaMalloc((int**)&d_result, NO_BYTES));

	// transform
	gpu_start = clock();
	gpuErrchk(cudaMemcpy(d_fir, h_fir, NO_BYTES, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sec, h_sec, NO_BYTES, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_thir, h_thir, NO_BYTES, cudaMemcpyHostToDevice));
	gpu_end = clock();
	get_runtime("GPU transformation", (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC), block_size);

	// launch grid
	dim3 block(block_size);
	dim3 grid(size / block.x);

	gpu_start = clock();
	sum_three_array_gpu << <grid, block >> > (d_fir, d_sec,d_thir, d_result, size);
	cudaDeviceSynchronize();
	gpu_end = clock();
	get_runtime("GPU", (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC), block_size);

	// return result
	gpuErrchk(cudaMemcpy(gpu_result, d_result, NO_BYTES, cudaMemcpyDeviceToHost));

	// array comparison: effective check
	compare_arrays(gpu_result, cpu_result, size);

	cudaFree(d_fir);
	cudaFree(d_sec);
	cudaFree(d_thir);
	cudaFree(d_result);


	free(h_fir);
	free(h_sec);
	free(h_thir);
	free(gpu_result);
	free(cpu_result);
	cudaDeviceReset();
}

// query device
void query_device()
{
	// 查询可用device数量
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) printf("No CUDA support device found\n"); else printf("There are %d available device(s)\n", deviceCount);

	// 查询设备，接受两个参数：CUDA设备属性类型变量以及设备编号，前者是函数返回到这个变量中，后者是要查询的目标设备。
	int devNo = 0;//我们仅查询第一个设备
	cudaDeviceProp iProp;
	cudaGetDeviceProperties(&iProp, devNo);

	printf("Device %d: %s\n", devNo, iProp.name);
	printf("\t Number of multiprocessors:\t %d \n", iProp.multiProcessorCount);
	printf("\t clock rate: \t %d \n", iProp.clockRate);
	printf("\t Compute capability: \t %d .%d \n", iProp.major, iProp.minor);
	printf("\t Total amount of global memory: \t %4.2f KB \n", iProp.totalGlobalMem / 1024.0);
	printf("\t Total amount of constant memory: \t %4.2f KB \n", iProp.totalConstMem / 1024.0);
	printf("\t Total amount of shared memory per block: \t %4.2f KB \n", iProp.sharedMemPerBlock / 1024.0);
	printf("\t Total amount of shared memory per MP: \t %4.2f KB \n", iProp.sharedMemPerMultiprocessor / 1024.0);
	printf("\t Total number of registers available per block: \t %d \n",iProp.regsPerBlock);
	printf("\t Warp size: \t %d \n",iProp.warpSize );
	printf("\t Maximum number of threads per block: \t %d \n", iProp.maxThreadsPerBlock);
	printf("\t Maximum number of threads per Multiprocessor: \t %d \n",iProp.maxThreadsPerMultiProcessor );
	printf("\t Maximum number of blocks per Multiprocessor: \t %d \n",iProp.maxBlocksPerMultiProcessor );
	printf("\t Maximum Grid size: \t (%d,%d,%d) \n",iProp.maxGridSize[0], iProp.maxGridSize[1], iProp.maxGridSize[2]);
	printf("\t Maximum block dimension: \t (%d,%d,%d) \n", iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);

}

// print details of warps
__global__ void print_details_of_warps()
{
	// 以一个grid(n,m,1), block(k,1,1)的分配为例
	int globalID = blockIdx.y * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id = threadIdx.x / 32;
	int globalblockId = gridDim.x * blockIdx.y + blockIdx.x;
	printf("tid : %d, bid.x : %d, bid.y : %d, gid : %d, warpid : %d, gbid: %d\n ", threadIdx.x, blockIdx.x, blockIdx.y, globalID, warp_id, globalblockId);
}

// test branch divergence
__global__ void code_without_divergence()
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	float a, b;
	a = b = 0;

	int warp_id = gid / 32;

	if (warp_id % 2 == 0)
	{
		a = 100.0;
		b = 50.0;
	}
	else
	{
		a = 200.0;
		b = 75.0;
	}
}
__global__ void divergence_code() {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	float a, b;
	a = b = 0;

	

	if (gid % 2 == 0)
	{
		a = 100.0;
		b = 50.0;
	}
	else
	{
		a = 200.0;
		b = 75.0;
	}
}

// occupancy test
__global__ void occupancy_test(int* result) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	int x1 = 1;
	int x2 = 2;
	int x3 = 3;
	int x4 = 4;
	int x5 = 5;
	int x6 = 6;
	int x7 = 7;
	int x8 = 8;
	
	result[gid] = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8;
}

// vector-add---------------------------------------selfcourse
__global__ void vec_add(float* x, float* y, float* z, int size)
{
	// under grid(s,s,1) block(256,1,1)
	int idx = (blockDim.x * gridDim.x * blockIdx.y) + (blockDim.x * blockIdx.x) + threadIdx.x;
	if (idx < size) z[idx] = x[idx] + y[idx];
}
void vec_add_cpu(float* x, float* y, float* z, int size)
{
	for (int i{ 0 }; i < size; i++)
	{
		z[i] = x[i] + y[i];
	}
}
// 向量加法的通常写法
/*
# define SIZEOFDATA 10000000
# define BLOCKSIZE 256

__global__ void mem_bw(float* A, float* B, float* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		float4 a1 = reinterpret_cast<float4*>(A)[idx];
		float4 b1 = reinterpret_cast<float4*>(B)[idx];
		float4 c1;
		c1.x = a1.x + b1.x;
		c1.y = a1.y + b1.y;
		c1.z = a1.z + b1.z;
		c1.w = a1.w + b1.w;

		reinterpret_cast<float4*>(C)[idx];
	}
}
int main() {
	int float4_array_size = (SIZEOFDATA + 4 - 1) / 4;
	dim3 block(BLOCKSIZE, 1, 1);
	dim3 grid((float4_array_size + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1);
	//略
	mem_bw << <grid, block >> > (A, B, C, float4_array_size);
	//略
}
*/


//int main() {
	/*
	dim3 block(8,2);
	dim3 grid(2,2);
	*/

	// Hello CUDA
	/*

	int nx{ 16 };
	int ny{ 4 };
	dim3 block(8, 2);
	dim3 grid(nx / block.x, ny / block.y);

	hello_cuda << <grid, block>> > ();

	cudaDeviceSynchronize();
	
	cudaDeviceReset();
	*/

	// thread唯一标识符
	/*

	int nx{ 16 };
	int ny{ 16 };

	dim3 block(8, 8);
	dim3 grid(nx / block.x, ny / block.y);
	print_threadIds << <grid, block >> > ();
	cudaDeviceSynchronize();
	cudaDeviceReset();
	*/

	// detail key words
	/*
	
	int nx{ 32 };
	int ny{ 16 };
	dim3 block(4, 4);
	dim3 grid(nx / block.x, ny / block.y);
	print_key_words_detail << <grid, block >> > ();
	cudaDeviceSynchronize();
	cudaDeviceReset();
	*/

	// 每个线程打印一个数据
	/*
	
	int array_size{ 8 };
	int array_byte_size=sizeof(int) * array_size;
	int h_data[] { 23,9,4,53,65,12,1,33 };

	for (auto i{ 0 }; i < array_size; i++)
	{
		printf("%d\t", h_data[i]);
	}
	printf("\n \n");

	int* d_data; 
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(8);
	dim3 grid(1);
	unique_idx_calc_threadIdx << <grid, block >> > (d_data);
	cudaDeviceSynchronize();
	cudaDeviceReset();
	*/

	// 计算全局唯一标识
	/*
	int array_size = 16;
	size_t array_byte_size{ sizeof(int) * array_size };
	int h_data[]{ 23,9,4,53,65,12,1,33,87,45,23,12,342,56,44,99 };

	for (size_t i{ 0 }; i < array_size; i++) {
		printf("%d ", h_data[i]);
	}

	printf("\n \n");

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block{ 4 };
	dim3 grid{ 4 };

	unique_gid_calc_threadIdx << <grid, block >> > (d_data);
	cudaDeviceSynchronize();
	cudaDeviceReset();
	*/

	// 内存传输测试
	/*
	int size = 128; 
	size_t byte_size = size * sizeof(int);

	// 指向整数数组的指针，此变量是host变量。
	int * h_input;
	// 分配内存: provide number of bytes we need to allocate as the argument to the malloc function
	h_input = (int*)malloc(byte_size);// malloc function will return a void pointer, so we have to cast that pointer to the type we want

	// 生成测试数组
	time_t t;
	srand((unsigned)time(&t));
	for (size_t i{ 0 }; i < size; i++)
	{
		//分配0-255随机值
		h_input[i] = (int)(rand() & 0xff);
	}

	// 指向设备的指针，并分配device中的内存
	int * d_input;
	// 采用double pointer(pointer to a pointer)作为第一个参数，因此我们需要先将设备指针转换为通用双指针(generic double pointer)，然后指定所需空间大小
	cudaMalloc((void**)&d_input, byte_size);

	// 传输数据
	cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

	// 设置启动参数
	dim3 block(64);
	dim3 grid(2);

	memory_transfer_test << <grid, block >> > (d_input);
	cudaDeviceSynchronize();
	
	// 回收Host和Device分配的内存
	cudaFree(d_input);
	free(h_input);

	cudaDeviceReset();
	 */
	
	// array summury and 有效性测试（GPU结果是否正确）
	/*
	int size = 10000;
	int block_size = 128;

	int NO_BYTES = (int)(size * sizeof(int));

	//hostpointer
	int* h_a, * h_b, * gpu_results, *h_c;
	h_a = (int*)malloc(NO_BYTES);
	h_b = (int*)malloc(NO_BYTES);
	gpu_results = (int*)malloc(NO_BYTES);
	h_c = (int*)malloc(NO_BYTES);

	//initialize host pointer
	time_t t;
	srand((unsigned int)time(&t));
	for (size_t i{ 0 }; i < size; i++)
	{
		h_a[i] = (int)(rand() & 0xFF);
	}
	for (size_t i{ 0 }; i < size; i++)
	{
		h_b[i] = (int)(rand() & 0xFF);
	}
	sum_array_cpu(h_a, h_b, h_c, size);

	memset(gpu_results, 0, NO_BYTES);

	// device pointer
	int* d_a, * d_b, * d_c;
	cudaMalloc((int**)&d_a, NO_BYTES);
	cudaMalloc((int**)&d_b, NO_BYTES);
	cudaMalloc((int**)&d_c, NO_BYTES);

	// transform
	cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);

	// launch grid
	dim3 block(block_size);
	dim3 grid((size / block.x) + 1);// 1000/128除不尽，加上一个block

	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, size);
	cudaDeviceSynchronize();

	// 传回结果
	cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);

	// array comparison: effective check
	compare_arrays(gpu_results, h_c, size);

	cudaFree(d_c);
	cudaFree(d_a);
	cudaFree(d_b);


	free(gpu_results);
	free(h_a);
	free(h_b);
	cudaDeviceReset();
	*/
	
	// sum array example with error check
	/*
	int size = 10000;
	int block_size = 128;
	cudaError error; // 在所有cuda函数调用之前都可使用

	int NO_BYTES = (int)(size * sizeof(int));

	//hostpointer
	int* h_a, * h_b, * gpu_results, * h_c;
	h_a = (int*)malloc(NO_BYTES);
	h_b = (int*)malloc(NO_BYTES);
	gpu_results = (int*)malloc(NO_BYTES);
	h_c = (int*)malloc(NO_BYTES);

	//initialize host pointer
	time_t t;
	srand((unsigned int)time(&t));
	for (size_t i{ 0 }; i < size; i++)
	{
		h_a[i] = (int)(rand() & 0xFF);
	}
	for (size_t i{ 0 }; i < size; i++)
	{
		h_b[i] = (int)(rand() & 0xFF);
	}
	sum_array_cpu(h_a, h_b, h_c, size);

	memset(gpu_results, 0, NO_BYTES);

	// device pointer
	int* d_a, * d_b, * d_c;
	error = cudaMalloc((int**)&d_a, NO_BYTES);//例如此处就可以使用
	if (error != cudaSuccess) {
		fprintf(stderr, "Error %s \n", cudaGetErrorString(error));
	}
	cudaMalloc((int**)&d_b, NO_BYTES);
	cudaMalloc((int**)&d_c, NO_BYTES);

	// transform
	cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);

	// launch grid
	dim3 block(block_size);
	dim3 grid((size / block.x) + 1);// 1000/128除不尽，加上一个block

	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, size);
	cudaDeviceSynchronize();

	// 传回结果
	cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);

	// array comparison: effective check
	compare_arrays(gpu_results, h_c, size);

	cudaFree(d_c);
	cudaFree(d_a);
	cudaFree(d_b);


	free(gpu_results);
	free(h_a);
	free(h_b);
	cudaDeviceReset();
	*/
	
	//homework 2
	/*
	int block_size[]{ 1 << 5, 1 << 6,1 << 7,1 << 8 };
	for (size_t i{ 0 };i < 4; i++)
		execute_homework2(block_size[i]);
	*/
	
	// query device
	/*
	query_device();
	*/

	// print details of warps;
	/*
	dim3 block(42, 1, 1);
	dim3 grid(2, 2, 1);
	print_details_of_warps << <grid, block >> > ();
	cudaDeviceSynchronize();
	*/
	
	// divergence test;
	/*
	printf("\n----------------------WARP DIVERGENCE EXAMPLE-----------------------------------\n\n");

	int size = 1 << 22;
	dim3 block_size(128);
	dim3 grid_size((size + block_size.x - 1) / block_size.x);

	code_without_divergence << <grid_size, block_size >> > ();
	cudaDeviceSynchronize();

	divergence_code << <grid_size, block_size >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	*/
	
	// occupancy test
	/*
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);
	int* h_result;
	int* d_result;
	
	h_result = (int*)malloc(sizeof(int));
	cudaMalloc((void**)&d_result, (int)sizeof(int));
	occupancy_test << <grid, block >> > (d_result);
	cudaDeviceSynchronize();
	// 传输数据
	cudaMemcpy(d_result, h_result, sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(d_result);
	free(h_result);
	cudaDeviceReset();
	*/

	// vector add
	/*
	int size = 10000;
	int nbytes = size * sizeof(float);
	int block = 256;

	int s = ceil(sqrt((size + block - 1.) / block));
	dim3 grid(s, s, 1);

	float* dx, * hx, * dy, * hy, * dz, * hz;

	cudaMalloc((void**)&dx, nbytes);
	cudaMalloc((void**)&dy, nbytes);
	cudaMalloc((void**)&dz, nbytes);

	// init time
	float milliseconds = 0;

	hx = (float*)malloc(nbytes);
	hy = (float*)malloc(nbytes);
	hz = (float*)malloc(nbytes);

	for (int i = 0; i < size; i++)
	{
		hx[i] = 1;
		hy[i] = 1;
	}

	cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	vec_add << <grid, block >> > (dx, dy, dz, size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(hz, dz, nbytes, cudaMemcpyDeviceToHost);

	float* hz_cpu_res = (float*)malloc(nbytes);
	vec_add_cpu(hx, hy, hz_cpu_res, size);

	for (int i = 0; i < size; ++i) {
		if (fabs(hz_cpu_res[i] - hz[i]) > 1e-6) {
			printf("Result verification failed at element index %d!\n", i);
		}
	}
	printf("Result right\n");
	printf("Mem BW= %f (GB/sec)\n", (float)size * 4 / milliseconds / 1e6);///1.78gb/s
	cudaFree(dx);
	cudaFree(dy);
	cudaFree(dz);

	free(hx);
	free(hy);
	free(hz);
	free(hz_cpu_res);
	*/


//	return 0;
//}


