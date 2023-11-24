# include<stdio.h>
# include<cuda.h>
# include"cuda_runtime.h"
# include<iostream>
# include<math.h>
# include<string>
# include<vector>
using namespace std;

// baseline 
__global__ void reduce_baseline(const int* input, int* output, size_t n) {
	int sum = 0;
	for (size_t i = 0; i < n; i++) {
		sum += input[i];
	}
	*output = sum;
}

bool CheckResult(int* out, int groudtruth, int n) {
	if (*out != groudtruth) {
		return false;
	}
	return true;
}

// reduce_v0
template<int blockSize>
__global__ void reduce_v0(float* d_in, float* d_out) {
	//存入共享内存中
	__shared__ float smem[blockSize];

	int tid = threadIdx.x;
	// int gtid = blockIdx.x * blockSize + threadIdx.x;
	// 每一个线程加载一个元素到shared memory对应的位置
	smem[tid] = d_in[tid];
	//等到所有thread全部完成以上操作后才继续
	__syncthreads();

	// 在shared memory中reduce
	for (int index = 1; index < blockDim.x; index *= 2) {
		if (tid % (2 * index) == 0) {
			smem[tid] += smem[tid + index];
		}
		__syncthreads();
	}

	// 最后全部存到0位置，这就是这个block的结果
	if (tid == 0) {
		d_out[blockIdx.x] = smem[0];
	}
}

bool CheckResult(float* out, float groudtruth, int n) {
	float res = 0;
	for (int i = 0; i < n; i++) {
		res += out[i];
	}

	if (res != groudtruth) {
		return false;
	}
	return true;
}


//int main(int argc, char** argv) {
//	/*
//	属性-命令参数-输入 1D baseline即可，这样argv[0]是这个程序的绝对路径，argv[1]是1D,argv[2]是baseline
//	argc = 3
//	argv[1] = "1D" or "2D"
//	argv[2] = "baseline", "v0", 
//	*/
//	vector<string> args(argv, argv + argc);
//	/*
//	cout << argc << endl;
//	cout << args[0] << endl;
//	cout << args[1] << endl;
//	cout << args[2] << endl;
//	*/
//	
//	if (args[1] == "1D" && args[2] == "baseline") {
//		/*
//		baseline
//		用一个thread向global memory读取N次，加完所有的数
//		*/
//		float milliseconds = 0;
//		cudaSetDevice(0);
//		cudaDeviceProp deviceProp;
//		cudaGetDeviceProperties(&deviceProp, 0);
//		const int N = 25600000;
//		const int blockSize = 1;
//		const int GridSize = 1;
//
//		//申请内存
//		//输入一个大小为N的数组
//		int* a = (int*)malloc(N * sizeof(int));
//		int* d_a;
//		cudaMalloc((void**)&d_a, N * sizeof(int));
//		//输出的是一个grid中的每个block计算的结果，存为一个数组
//		int* out = (int*)malloc((GridSize) * sizeof(int));
//		int* d_out;
//		cudaMalloc((void**)&d_out, (GridSize) * sizeof(int));
//
//
//		for (int i = 0; i < N; i++) {
//			a[i] = 1;
//		}
//
//		int groudtruth = N * 1;
//
//		cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
//
//		dim3 Grid(GridSize);
//		dim3 Block(blockSize);
//
//		cudaEvent_t start, stop;
//		cudaEventCreate(&start);
//		cudaEventCreate(&stop);
//		cudaEventRecord(start);
//		reduce_baseline << <1, 1 >> > (d_a, d_out, N);
//		cudaEventRecord(stop);
//		cudaEventSynchronize(stop);
//		cudaEventElapsedTime(&milliseconds, start, stop);
//
//		cudaMemcpy(out, d_out, GridSize * sizeof(int), cudaMemcpyDeviceToHost);
//		printf("allcated %d blocks, data counts are %d", GridSize, N);
//
//		bool is_right = CheckResult(out, groudtruth, GridSize);
//		if (is_right) {
//			printf("the ans is right\n");
//		}
//		else {
//			printf("the ans is right\n");
//			for (int i = 0; i < GridSize; i++) {
//				printf("res per block : %lf", out[i]);
//			}
//			printf("\n");
//			printf("groudtruth is %f \n", groudtruth);
//		}
//		printf("reduce_baseline latency = %f ms\n", milliseconds);
//
//		cudaFree(d_a);
//		cudaFree(d_out);
//		free(a);
//		free(out);
//
//	}
//	else if (args[1] == "1D" && args[2] == "v0") {
//		/*
//		1. 使用每个thread先加载一个数据到其shared memory中
//		2. thread两两相加，利用__syncthreads同步所有thread的行为，如此若干轮
//		3. 最后一个block的结果会存在这个block的thread0中
//		*/
//		float milliseconds = 0;
//		cudaSetDevice(0);
//		cudaDeviceProp deviceProp;
//		cudaGetDeviceProperties(&deviceProp, 0);
//		const int N = 25600000;
//		const int blockSize = 256;
//		int GridSize = min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
//		//申请内存
//		//输入一个大小为N的数组
//		float* a = (float*)malloc(N * sizeof(float));
//		float* d_a;
//		cudaMalloc((void**)&d_a, N * sizeof(float));
//		//输出的是一个grid中的每个block计算的结果，存为一个数组
//		float* out = (float*)malloc(GridSize * sizeof(float));
//		float* d_out;
//		cudaMalloc((void**)&d_out, GridSize * sizeof(float));
//
//		// 初始化输入数组为全1
//		for (int i = 0; i < N; i++) {
//			a[i] = 1.0f;
//		}
//		// 答案应该就等于数组大小
//		float groudtruth = N * 1.0f;
//		// 传给GPU
//		cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
//
//		dim3 Grid(GridSize);
//		dim3 Block(blockSize);
//
//		cudaEvent_t start, stop;
//		cudaEventCreate(&start);
//		cudaEventCreate(&stop);
//		cudaEventRecord(start);
//		reduce_v0 <blockSize> <<<Grid, Block >> > (d_a, d_out);
//		cudaEventRecord(stop);
//		cudaEventSynchronize(stop);
//		cudaEventElapsedTime(&milliseconds, start, stop);
//
//		cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
//		printf("allcated %d blocks, data counts are %d \n", GridSize, N);
//
//		bool is_right = CheckResult(out, groudtruth, GridSize);
//		if (is_right) {
//			printf("the ans is right\n");
//		}
//		else {
//			printf("the ans is wrong\n");
//			for (int i = 0; i < GridSize; i++) {
//				printf("res per block : %lf\n", out[i]);
//			}
//			printf("\n");
//			printf("groudtruth is %f \n", groudtruth);
//		}
//		printf("reduce_v0 latency = %f ms\n", milliseconds);
//
//		cudaFree(d_a);
//		cudaFree(d_out);
//		free(a);
//		free(out);
//	}
//	else {
//		cout << " 没有一个符合" << endl;
//		return 0;
//	}
//	
//
//
//	
//
//	return 0;
//}


