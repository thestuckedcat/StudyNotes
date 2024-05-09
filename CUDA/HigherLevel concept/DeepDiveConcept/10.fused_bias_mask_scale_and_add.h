#pragma once
#include<stdio.h>
#include<cuda.h>
#include"cuda_runtime.h"

/*
	这个做的是，考虑一个输入是[batch_size,input]的大小
	它经过了一个MLP，变成了
	[batch_size,input]*[input,output] = [batch_size,output]
	这个[batch_size,output]大小的数据我们用float* x传递

	它因为经过了MLP，需要一个bias，bias的大小应该为[output]

	在这个算子中，主要做了以下工作

	1. 对batch_size*output大小的 x数组每[output]个元素加上bias，完成MLP

	2. 对batch_size个output元素添加mask

	考虑层归一化
	y = \gamma * [(x-\mu)/ \sqrt(\sigma^2 + \epsilon)] + \beta

	3. 归一化，为所有元素缩放(\gamma)，调整比例，此处为LayerNorm,单个值

	4. 归一化中，通常会有一个逐元素的偏置(\beta),长度为元素个数


*/
void test_fp32_fused_kernel();

void test_fp16_fused_kernel();

