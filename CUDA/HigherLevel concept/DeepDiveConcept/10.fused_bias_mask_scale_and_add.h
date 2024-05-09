#pragma once
#include<stdio.h>
#include<cuda.h>
#include"cuda_runtime.h"

/*
	��������ǣ�����һ��������[batch_size,input]�Ĵ�С
	��������һ��MLP�������
	[batch_size,input]*[input,output] = [batch_size,output]
	���[batch_size,output]��С������������float* x����

	����Ϊ������MLP����Ҫһ��bias��bias�Ĵ�СӦ��Ϊ[output]

	����������У���Ҫ�������¹���

	1. ��batch_size*output��С�� x����ÿ[output]��Ԫ�ؼ���bias�����MLP

	2. ��batch_size��outputԪ�����mask

	���ǲ��һ��
	y = \gamma * [(x-\mu)/ \sqrt(\sigma^2 + \epsilon)] + \beta

	3. ��һ����Ϊ����Ԫ������(\gamma)�������������˴�ΪLayerNorm,����ֵ

	4. ��һ���У�ͨ������һ����Ԫ�ص�ƫ��(\beta),����ΪԪ�ظ���


*/
void test_fp32_fused_kernel();

void test_fp16_fused_kernel();

