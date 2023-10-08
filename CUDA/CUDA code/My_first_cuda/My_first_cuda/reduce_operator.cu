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
	//���빲���ڴ���
	__shared__ float smem[blockSize];

	int tid = threadIdx.x;
	// int gtid = blockIdx.x * blockSize + threadIdx.x;
	// ÿһ���̼߳���һ��Ԫ�ص�shared memory��Ӧ��λ��
	smem[tid] = d_in[tid];
	//�ȵ�����threadȫ��������ϲ�����ż���
	__syncthreads();

	// ��shared memory��reduce
	for (int index = 1; index < blockDim.x; index *= 2) {
		if (tid % (2 * index) == 0) {
			smem[tid] += smem[tid + index];
		}
		__syncthreads();
	}

	// ���ȫ���浽0λ�ã���������block�Ľ��
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
//	����-�������-���� 1D baseline���ɣ�����argv[0]���������ľ���·����argv[1]��1D,argv[2]��baseline
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
//		��һ��thread��global memory��ȡN�Σ��������е���
//		*/
//		float milliseconds = 0;
//		cudaSetDevice(0);
//		cudaDeviceProp deviceProp;
//		cudaGetDeviceProperties(&deviceProp, 0);
//		const int N = 25600000;
//		const int blockSize = 1;
//		const int GridSize = 1;
//
//		//�����ڴ�
//		//����һ����СΪN������
//		int* a = (int*)malloc(N * sizeof(int));
//		int* d_a;
//		cudaMalloc((void**)&d_a, N * sizeof(int));
//		//�������һ��grid�е�ÿ��block����Ľ������Ϊһ������
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
//		1. ʹ��ÿ��thread�ȼ���һ�����ݵ���shared memory��
//		2. thread������ӣ�����__syncthreadsͬ������thread����Ϊ�����������
//		3. ���һ��block�Ľ����������block��thread0��
//		*/
//		float milliseconds = 0;
//		cudaSetDevice(0);
//		cudaDeviceProp deviceProp;
//		cudaGetDeviceProperties(&deviceProp, 0);
//		const int N = 25600000;
//		const int blockSize = 256;
//		int GridSize = min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
//		//�����ڴ�
//		//����һ����СΪN������
//		float* a = (float*)malloc(N * sizeof(float));
//		float* d_a;
//		cudaMalloc((void**)&d_a, N * sizeof(float));
//		//�������һ��grid�е�ÿ��block����Ľ������Ϊһ������
//		float* out = (float*)malloc(GridSize * sizeof(float));
//		float* d_out;
//		cudaMalloc((void**)&d_out, GridSize * sizeof(float));
//
//		// ��ʼ����������Ϊȫ1
//		for (int i = 0; i < N; i++) {
//			a[i] = 1.0f;
//		}
//		// ��Ӧ�þ͵��������С
//		float groudtruth = N * 1.0f;
//		// ����GPU
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
//		cout << " û��һ������" << endl;
//		return 0;
//	}
//	
//
//
//	
//
//	return 0;
//}


