
# include"9.gelu.h"
# include<cassert>
# include"utils.h"
# include<cmath>
# include<iostream>

/*
* 
* // 以下妄图使用PTX指令被薄纱，望周知
__device__ __half GeluFunctor<__half>::operator()(__half x) const {
	__half tanh_in = alpha_half * (x + beta_half * x * x * x);
	__half tanh_out;

	// PTX ISA documentation
	asm("tanh.approx.f16 %0,%1;\n\t":"=h"(tanh_out) : "h"(tanh_in));

	return __float2half_rn(0.5f) * x * (__float2half_rn(1.0f) + tanh_out);
}


__device__ void GeluFunctor<__half>::vec_gelu(__half* output, const __half* input) const {
	// 从half数组读取一个half2的值
	__half2 x2 = *(reinterpret_cast<const __half2*> (input));

	// half2直接计算
	__half2 tanh_in = __half2half2(alpha_half) * (x2 + __half2half2(beta_half) * x2 * x2 * x2);

	__half2 tanh_out;

	asm("tanh.approx.f16x2 %0,%1; \n\t": "=f"(tanh_out) : "f"(tanh_in));

	__half2 output2 = *(reinterpret_cast<const __half2*>(output));

	output2 = __half2half2(__float2half_rn(0.5f)) * x2 * (__half2half2(__float2half_rn(1.0f)) + tanh_out);
}

__device__ __half GeluFunctor<__half>::gelu_f_accuracy(__half x) const
{
	const float tanh_in = alpha_f * (__half2float(x) + beta_f * __half2float(x) * __half2float(x) * __half2float(x));

	float tanh_out;

	asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(tanh_out) : "f"(tanh_in));

	return __float2half(0.5f * __half2float(x) * (1.0f + tanh_out));
}


template<int VecSize>
__global__ void FP16GeluCUDAKernel(const __half* x, __half* y, int n){
	// 向量化load & store, x=input, y=output
	// 读取向量的offset

	// offset of __half
	int offset = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) * VecSize;

	// 循环读取向量的stride
	int stride = static_cast<int>(blockDim.x * gridDim.x) * VecSize;

	GeluFunctor<__half> gelu_forward;

	__half output_local[VecSize];

	// 循环处理
	for (; offset < n; offset += stride) {
		using AV = AlignedVector<__half, VecSize>;
		// vec's address of this 
		const AV* in_arr = reinterpret_cast<const AV*>(x + offset);


		const __half* in = reinterpret_cast<const __half*>(in_arr);
		if (VecSize == 1) {
			// 以单个half计算
			// output_local[0] = gelu_forward(in[0]);
			output_local[0] = gelu_forward.gelu_f_accuracy(in[0]);
		}
		else if (VecSize == 2) {

			//向量化__half2计算
			gelu_forward.vec_gelu(output_local, in);
		}
	}

	// 将值复制到y中传出kernel
	// 这里必须用赋值，不能直接将地址赋予，即必须deep copy
	*reinterpre_cast<AV*> (y + offset) = *reinterpret_cast<AV*>(output_local);


}

// gelu公式：x / 2 * (1 + tanh(0.7978845608028654 * (x + 0.044714998453855515 * x^3)))
void cpu_compute_result(int n, const __half* x, __half* y) {
	
	for (int i = 0; i < n; i++) {
		float tanh_in = __half2float(x[i]);
		tanh_in = tanh_in + (float)0.044714998453855515 * tanh_in * tanh_in * tanh_in;
		tanh_in *= (float)0.7978845608028654;
		float tanh_out = tanh(tanh_in);

		y[i] = __float2half((tanh_out + 1) * 0.5 * __half2float(x[i]));
	}
}

void CHECK_Result(__half* h_res, __half* d_res, const int n) {
	for (int i = 0; i < n; i++) {
		if (__habs(h_res[i] - d_res[i]) > __double2half(1e-5)) {
			std::cout << "CPU result and GPU result does not match on " << i << "CPU:\t" << __half2float(h_res[i]) << "\t GPU:\t" << __half2float(d_res[i]) << std::endl;
		}
	}
}

void Gelu_FP16_kernellauncher() {
	constexpr int n = 2048;
	const int vecsize = 2;
	__half* x = (__half*)malloc(sizeof(__half) * n);
	__half* y_d_out = (__half*)malloc(sizeof(__half) * n);
	__half* y_h_out = (__half*)malloc(sizeof(__half) * n);

	for (int i = 0; i < n; i++) {
		x[i] = (__half)(i);
	}

	cpu_compute_result(n, x, y_h_out);

	__half* d_x, * d_y;

	cudaMalloc((void**)&d_x, n * sizeof(__half));
	cudaMalloc((void**)&d_y, n * sizeof(__half));
	cudaMemcpy(d_x, x, sizeof(__half) * n, cudaMemcpyHostToDevice);


	// CHECK if block % VecSize == 0 && n % block == 0, no bound check is applied in kernel
	
	int block_size = 512;
	int grid_size = 2;
	assert(n % vecsize == 0 && block_size % vecsize == 0);
	FP16GeluCUDAKernel<vecsize><< <grid_size, block_size>> >(d_x, d_y, n) ;

	cudaMemcpy(y_h_out, d_y, sizeof(__half) * n, cudaMemcpyDeviceToHost);


	CHECK_Result(y_h_out, y_d_out, n);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y_h_out);
	free(y_d_out);

}

*/
__device__ float SelfTanhApprox(float x) {
	// ptx指令，是CUDA的更底层的语言，类似于汇编对于C/C++
	//float r;
	//asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
	//return r;
	return tanhf(x); // CUDA内置的math API
}

void test() {
	int n = 1000;

	__half* x = new __half[n];
	__half* y = new __half[n];
	for (int i = 0; i < n; i++)
	{
		x[i] = (__half)(i);
	}
	__half* d_x, * d_y;
	cudaMalloc((void**)&d_x, n * sizeof(__half));
	cudaMalloc((void**)&d_y, n * sizeof(__half));
	cudaMemcpy(d_x, x, sizeof(__half) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, sizeof(__half) * n, cudaMemcpyHostToDevice);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	auto is_aligned = [](const void* p, int alignment) {
		return reinterpret_cast<uintptr_t>(p) % alignment == 0;
		};

	constexpr auto kAlignment = alignof(AlignedVector<__half, 8>);
	// Note: when you have ampere GPU, you can enable the 122-124 line to get performance improvement by half2 intrinsic.
	if (n % 8 == 0 && is_aligned(x, kAlignment) && is_aligned(y, kAlignment)) {
		int thread = std::min<int>(512, deviceProp.maxThreadsPerBlock);
		//int block = (n / 8 + thread - 1) / thread;                  
		//block = std::min<int>(block, deviceProp.maxGridSize[0]);                                  
		//FP16GeluCUDAKernel<8, true><<<block, thread>>>(x, y, n);  
		int block = (n + thread - 1) / thread;
		block = std::min<int>(block, deviceProp.maxGridSize[0]);
		FP16GeluCUDAKernel<1> << <block, thread >> > (d_x, d_y, n);
		cudaMemcpy(y, d_y, sizeof(__half) * n, cudaMemcpyDeviceToHost);
	}
	printf("pass");
	delete x;
	x = nullptr;
	delete y;
	y = nullptr;
	cudaFree(d_x);
	cudaFree(d_y);
}