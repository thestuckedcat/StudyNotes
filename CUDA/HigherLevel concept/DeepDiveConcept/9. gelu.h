#pragma once
# include<stdio.h>
# include<cuda.h>
# include<cuda_fp16.h>
# include"cuda_runtime.h"

template<typename T, int Size>
struct alignas(sizeof(T)* Size) AlignedVector {
	// 相当于一个不同类型的size 的aligned wrapper
	// 注意这里的T不能是char（虽然一般也不可能），并且这里的alignas稍显多余


	T val[Size];

	// __host__ __device__代表同时可以在CPU和GPU上调用和执行
	__host__ __device__ inline const T& operator[](int i) const {
		return val[i];
	}

	__host__ __device__ inline T& operator[](int i) {
		return val[i];
	}
};


__device__ float TanhApprox(float x) {
	// 计算Tanh的近似值
	//float r;
	// //%0:输出占位符，r，%1 输入zhan,x
	// //\n:换行，用于分隔指令，使得生成的汇编代码更清晰
	// // \t：制表符，用于缩进，增加代码可读性
	//asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
	//return r;
	return tanhf(x); //cuda math API
}



// gelu公式：x / 2 * (1 + tan(0.7978845608028654 * (x + 0.044714998453855515 * x^3)))

template<typename T>
struct GeluFunctor {
	static constexpr T alpha = static_cast<T>(0.7978845608028654);
	static constexpr T beta = static_cast<T>(0.044714998453855515);

	__device__ GeluFunctor(){}

	__device__ T operator()(T x) const {
		const T halff = static_cast<T>(0.5);
		const T one = static_cast<T>(1);
		const T tanh_in = alpha * (x + beta * x * x * x);
		return halff * x * (one + tanhf(tanh_in));
	}
};


template<>
struct GeluFunctor<__half> {
	// 计算时使用half
	__half alpha_half = __double2half(0.7978845608028654);
	__half beta_half = __double2half(0.044714998453855515); 


	// 计算时使用float,因为half精确度和表达范围小，不适合进行复杂的运算，例如指数对数
	static constexpr float alpha_f = GeluFunctor<float>::alpha;
	static constexpr float beta_f = GeluFunctor<float>::beta;


	__device__ GeluFunctor() {};

	// 计算时使用half
	__device__ __half operator()(__half x) const {
		__half tanh_in = alpha_half * (x + beta_half * x * x * x);
		__half tanh_out;

		// PTX ISA documentation
		asm("tanh.approx.f16 %0,%1; \n\t":"=f"(tanh_out) : "f"(tanh_in));

		return __float2half_rn(0.5f) * x * (__float2half_rn(1.0f) + tanh_out);
	}

	// 计算时使用half2
	__device__ void vec_gelu(__half* y, const __half* x) const {
		const __half2 x2 = *(reinterpret_cast<const __half2*> (x));
	}

};

