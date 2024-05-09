#include"10.fused_bias_mask_scale_and_add.h"
# include<cstdint>// uint8_t
# include<iostream>
#include<cuda_fp16.h>
# include<cassert>
template<typename T>
struct MaskAndNormFunctor {
	// mask
	const uint8_t* mask;
	// Norm calculate
	const T* add_val;
	float scale;
	MaskAndNormFunctor(const uint8_t* mask, const T* add_val, float scale)
		:mask(mask), add_val(add_val),scale(scale){}

	__device__ T Compute(T x, int64_t i) const {
		return x * static_cast<T>(static_cast<bool>(mask[i])*scale) + add_val[i];
	}


};
template<typename FUNCTOR, typename T>
__global__ void FusedBiasAddCUDAKernelFloat(
	FUNCTOR functor,
	const int elem_cnt,
	const int bias_size,
	const T* x,
	const T* bias,
	T* y) 
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int KernelSize = blockDim.x * gridDim.x;
	for (int i = tid; i < elem_cnt; i += KernelSize) {
		// Add MLP bias
		T x_i = x[i] + bias[i % bias_size];
		y[i] = functor.Compute(x_i, i);
	}
}

template<typename T>
void CPU_fused_kernel(
	uint8_t* mask, 
	T* add_val, 
	float scale, 
	T* cpu_input,
	int ele_cnt,
	T* bias,
	int bias_size,
	T* output) 
{

	for (int i = 0; i < ele_cnt; i++) {
		T x_i = cpu_input[i] + bias[i % bias_size];
		output[i] = x_i * static_cast<T>(static_cast<bool>(mask[i]) * scale) + add_val[i];
	}

}
template<typename T>
bool CHECK_RES(T* cpu_output, T* gpu_output, const int ele_cnt) {
	for (int i = 0; i < ele_cnt; i++) {
		if (std::abs(cpu_output[i] - gpu_output[i]) > (T)(1e-3)) {
			std::cout << "第" << i << "个元素出错" << std::endl;
			std::cout << cpu_output[i] << " " << gpu_output[i] << std::endl;
			return false;
		}
	}
	std::cout << "所有元素都一样" << std::endl;
	return true;
}

void test_fp32_fused_kernel() {
	constexpr int ele_cnt = 100000;
	float scale = 0.5;

	// parameter in cpu
	uint8_t* mask_tensor = new uint8_t[ele_cnt];
	float* add_val = new float[ele_cnt];
	for (int i = 0; i < ele_cnt; i++) {
		mask_tensor[i] = (uint8_t)(i%2);
		add_val[i] = (float)(i);
	}

	// bias,input,output in cpu
	int bias_size = 100;
	float* x = (float*)malloc(sizeof(float) * ele_cnt);
	float* y = (float*)malloc(sizeof(float) * ele_cnt);
	float* bias = (float*)malloc(sizeof(float) * bias_size);
	for (int i = 0; i < ele_cnt; i++) {
		x[i] = (float)(i);
	}
	for (int i = 0; i < bias_size; i++) {
		bias[i] = (float)(i);
	}

	// cpu_output
	float* cpu_output = (float*)malloc(sizeof(float) * ele_cnt);
	CPU_fused_kernel<float>(mask_tensor, add_val, scale, x, ele_cnt, bias, bias_size, cpu_output);


	// bias,input,output in gpu
	float* d_x, * d_y, * d_bias;
	cudaMalloc((void**)&d_x, ele_cnt * sizeof(float));
	cudaMalloc((void**)&d_y, ele_cnt * sizeof(float));
	cudaMalloc((void**)&d_bias, bias_size * sizeof(float));

	cudaMemcpy(d_x, x, sizeof(float) * ele_cnt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, sizeof(float) * ele_cnt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias, bias, sizeof(float) * bias_size, cudaMemcpyHostToDevice);

	// mask_tensor, add_cal in gpu
	uint8_t* d_mask_tensor;
	float* d_add_val;
	cudaMalloc((void**)&d_mask_tensor, ele_cnt * sizeof(uint8_t));
	cudaMalloc((void**)&d_add_val, ele_cnt* sizeof(float));
	cudaMemcpy(d_mask_tensor, mask_tensor, sizeof(uint8_t) * ele_cnt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_add_val, add_val, sizeof(float) * ele_cnt, cudaMemcpyHostToDevice);

	/*
		这是一个 CUDA 提供的结构体，用于存储 GPU 设备的各项属性。
		其中包含了 GPU 设备的名称、计算能力、内存大小、最大线程数、最大网格和块大小等信息。
	*/
	cudaDeviceProp deviceProp;
	// 将指定 GPU 设备的属性信息填充到 cudaDeviceProp 结构体中。
	cudaError_t message_GPU = cudaGetDeviceProperties(&deviceProp, 0);

	int maxblocks = deviceProp.maxGridSize[0];
	
	int blockSize = 1024;
	int gridSize = std::min((ele_cnt + blockSize - 1) / blockSize, maxblocks);

	MaskAndNormFunctor<float> MNF(d_mask_tensor, d_add_val, scale);

	FusedBiasAddCUDAKernelFloat << <gridSize, blockSize >> > (	MNF,
																ele_cnt,
																bias_size,
																d_x,
																d_bias,
																d_y);

	cudaMemcpy(y, d_y, sizeof(float) * ele_cnt, cudaMemcpyDeviceToHost);


	CHECK_RES(cpu_output, y, ele_cnt);

	delete[] mask_tensor;
	delete[] add_val;
	free(x);
	free(y);
	free(bias);
	free(cpu_output);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_bias);
	cudaFree(d_mask_tensor);
	cudaFree(d_add_val);





}
// fp16偏特例化
template<>
struct MaskAndNormFunctor<__half> {
	// mask
	const uint8_t* mask;
	// Norm calculate
	const __half* add_val;
	float scale;
	MaskAndNormFunctor(const uint8_t* mask, const __half* add_val, float scale)
		:mask(mask), add_val(add_val), scale(scale) {}

	// half compute
	__device__ __half Compute(__half x, int64_t i) const {
		return x * static_cast<half>(static_cast<bool>(mask[i]) * scale) + add_val[i];
	}

	//half2 compute
	__device__ __half2 VecCompute(__half2 x, int64_t i) const {

		// mask的uint8_t没有向量化数据结构，使用考虑一个char是8位，因此用以代替，这是不传入位置的tricky写法
		const char2* mask_vec = reinterpret_cast<const char2*>(mask);

		const __half2* add_val_vec = reinterpret_cast<const __half2*> (add_val);

		char2 mask_val = mask_vec[i];//向量化读取
		//转换为__half进行向量计算
		__half2 one_or_zero;
		one_or_zero.x = mask_val.x;
		one_or_zero.y = mask_val.y;
		__half2 scale_vec = __float2half2_rn(scale);


		//__hmul2(x, one_or_zero), scale_vec)									: mask过程
		//__hmul2(__hmul2(x, one_or_zero), scale_vec)							：Norm-scale过程
		//__hadd2(__hmul2(__hmul2(x, one_or_zero), scale_vec), add_val_vec[i]);	：Norm-bias过程
		return __hadd2(__hmul2(__hmul2(x, one_or_zero), scale_vec), add_val_vec[i]);
	}


};

template<typename FUNCTOR>
__global__ void FusedBiasAddCUDAKernelFloat<FUNCTOR,__half>(
	FUNCTOR functor,
	const int elem_cnt,
	const int bias_size,
	const __half* x,
	const __half* bias,
	__half* y)
{
	const int h2_ele_cnt = elem_cnt / 2;
	assert(elem_cnt % 2 == 0);
	const int h2_bias_size = bias_size / 2;
	assert(elem_cnt % 2 == 0);
	const auto* x_h2 = reinterpret_cast<const __half2*>(x);
	const auto* bias_h2 = reinterpret_cast<const __half2*>(bias);
	auto* y_h2 = reinterpret_cast<__half2*>(y);


	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int kernelSize = blockDim.x * gridDim.x;
	for (int i = tid; i < h2_ele_cnt; i += kernelSize) {
		
		__half2 x_i = __hadd2(x_h2[i], bias_h2[i % h2_bias_size]);
		//y_h2[i] = functor.Compute(x_i, i);
		y_h2[i] = functor.VecCompute(x_i, i);
	}

}


void test_fp16_fused_kernel() {

}