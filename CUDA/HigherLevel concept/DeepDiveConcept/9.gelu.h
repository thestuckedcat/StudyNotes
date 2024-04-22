#pragma once
# include<stdio.h>
# include<cuda.h>
# include<cuda_fp16.h>
# include"cuda_runtime.h"


/*
* // ��ͼʹ��PTX���³�
template<typename T, int Size>
struct alignas(sizeof(T)* Size) AlignedVector {
	// �൱��һ����ͬ���͵�size ��aligned wrapper
	// ע�������T������char����Ȼһ��Ҳ�����ܣ������������alignas���Զ���
	// ��������������ǶԸ���vec���ʹ���һ��wrapper������half2,half4,float2�ȣ�����ͳһ��ʾ

	T val[Size];

	// __host__ __device__����ͬʱ������CPU��GPU�ϵ��ú�ִ��
	__host__ __device__ inline const T& operator[](int i) const {
		return val[i];
	}

	__host__ __device__ inline T& operator[](int i) {
		return val[i];
	}
};




// ��Ҫ�ڲ�ͬ�ļ��㸺�غ��ڴ���������²��Ըú���
// 1. ����Ϊ��������Ȼ��̫���ܣ����������������ж�)

// gelu��ʽ��x / 2 * (1 + tanh(0.7978845608028654 * (x + 0.044714998453855515 * x^3)))

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
	// ����ʱʹ��half
	__half alpha_half = __double2half(0.7978845608028654);
	__half beta_half = __double2half(0.044714998453855515); 


	// ����ʱʹ��float,��Ϊhalf��ȷ�Ⱥͱ�ﷶΧС�����ʺϽ��и��ӵ����㣬����ָ������
	static constexpr float alpha_f = GeluFunctor<float>::alpha;
	static constexpr float beta_f = GeluFunctor<float>::beta;


	__device__ GeluFunctor() {};

	// Ĭ�ϼ���ʱʹ��half
	__device__ __half operator()(__half x) const;

	// ����ʱʹ��half2
	__device__ void vec_gelu(__half* output, const __half* input) const;



	// halfתfloat����
	// gelu��ʽ��x / 2 * (1 + tanh(0.7978845608028654 * (x + 0.044714998453855515 * x^3)))
	__device__ __half gelu_f_accuracy(__half x) const;


};





template<int VecSize>
__global__ void FP16GeluCUDAKernel(const __half* x, __half* y, int n); 


void Gelu_FP16_kernellauncher();


void Gelu_FP16_kernellauncher();

*/ 


template <typename T, int Size>
struct alignas(sizeof(T)* Size) AlignedVector {
    // ������size������ΪT��Ԫ�����
    T val[Size];
    // ����֧��[]����
    __host__ __device__ inline const T& operator[](int i) const { return val[i]; }
    __host__ __device__ inline T& operator[](int i) { return val[i]; }
};

__device__ float SelfTanhApprox(float x);

// gelu��ʽ��x / 2 * (1 + tan(0.7978845608028654 * (x + 0.044714998453855515 * x^3))), �������Բ�
template<typename T>
struct GeluFunctor {
    static constexpr T alpha = static_cast<T>(0.7978845608028654);
    static constexpr T beta = static_cast<T>(0.044714998453855515);

    __device__ GeluFunctor() {};

    __device__ T operator()(T x) const {
        const T half = static_cast<T>(0.5);
        const T one = static_cast<T>(1);
        const T tanh_in = alpha * (x + beta * x * x * x);
        return half * x * (one + tanh(tanh_in));
    }
};

template<>
struct GeluFunctor<half> {
    // ͵�˸�����ֱ�Ӱ�L26��L27�ù�����
    static constexpr float alpha = GeluFunctor<float>::alpha;
    static constexpr float beta = GeluFunctor<float>::beta;
    GeluFunctor<float> float_functor;

    __device__ GeluFunctor() {};

    __device__ half operator()(const half x) const {
        // Note: when you have ampere GPU, you can enable the line45-50 method to get performance improvement by half intrinsic instead of static_cast half to fp32.
        //const float tanh_in =
        //    __half2float(__float2half_rn(alpha) * (x + __float2half_rn(beta) * x * x * x));
        //const float tanh_out = TanhApprox(tanh_in);
        //return __float2half_rn(0.5f) * x * (__float2half_rn(1.0f) + __float2half_rn(tanh_out));
        // Note: half to float will lose performance using static_cast, because static_cast will be compiled to more instructions than half intrinsic,
        // so you should better use half intrinsic when you have ampere GPU, you can enable 44-47 line
        return static_cast<half>(float_functor(static_cast<float>(x)));
    }
    // Note: when you have ampere GPU, you can enable the "apply2" method to get performance improvement by half2 intrinsic.
    //__device__ void Apply2(half* y, const half* x) const {
      //const half2 x2 = *(reinterpret_cast<const half2*>(x));
      //const float2 tanh_in = __half22float2(
       //   __hmul2(__float2half2_rn(alpha),
        //          __hadd2(x2, __hmul2(__hmul2(__hmul2(__float2half2_rn(beta), x2), x2), x2))));
      //float2 tanh_out;
      //tanh_out.x = TanhApprox(tanh_in.x);
      //tanh_out.y = TanhApprox(tanh_in.y);
      //const half2 y2 = __hmul2(__hmul2(__float2half2_rn(0.5F), x2),
      //                                 __hadd2(__float2half2_rn(1.0F), __float22half2_rn(tanh_out)));
      //*reinterpret_cast<half2*>(y) = y2;
    //}
};


template <int VecSize>
__global__ void FP16GeluCUDAKernel(const __half* x,
    __half* y,
    int n) {
    // ������load & store
    // ��ȡ������offset
    int offset =
        static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
    // ѭ����ȡ������stride
    int stride = static_cast<int>(blockDim.x * gridDim.x) * VecSize;
    GeluFunctor<half> gelu_fwd;
    __half y_reg[VecSize];
    for (; offset < n; offset += stride) {
        // ��ǿתΪ�������ٴ���offset��ȡ��Ӧ����
        using ArrT = AlignedVector<__half, VecSize>;
        const ArrT* in_arr = reinterpret_cast<const ArrT*>(x + offset);
        const __half* in = reinterpret_cast<const __half*>(in_arr);

        if (VecSize == 1) {
            y_reg[0] = gelu_fwd(in[0]);
        }
        else {
            // Note: when you have ampere GPU, you can enable the "apply2" method replacing L99-L101 to get performance improvement by half2 intrinsic do vector computation.
            //for (int i = 0; i < VecSize; i+=2) {
            //gelu_fwd.apply2(y + offset, in[i]);
            //��������
            for (int i = 0; i < VecSize; i++) {
                y_reg[i] = gelu_fwd(in[i]);
            }
        }
        // ��������д���Դ�
        *reinterpret_cast<ArrT*>(y + offset) = *reinterpret_cast<ArrT*>(y_reg);
    }
}

void test();