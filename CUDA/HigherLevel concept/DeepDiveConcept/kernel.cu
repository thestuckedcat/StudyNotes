
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
# include "copy_if.h"
#include <stdio.h>
# include<random>
# include<iostream>
# include<chrono>
# include "cublas_example.h"
# include "copy_if_test.h"
# include "9.gelu.h"
# include"10.fused_bias_mask_scale_and_add.h"




int main()
{	
	/*
	// copy_if example
	int* src = generate_random_array();
	compare1(src);
	compare2(src);
	compare3(src);
	delete[] src;
	*/


	////cublas example
	//int result = test_cublas();
	//std::cout << result << std::endl;


	//test();

	//test_fp32_fused_kernel();
	test_fp16_fused_kernel();

}