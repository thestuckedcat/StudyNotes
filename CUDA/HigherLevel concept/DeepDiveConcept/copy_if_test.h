#pragma once
#include "utils.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
# include "copy_if.h"
#include <stdio.h>
# include<random>
# include<iostream>
# include<chrono>
# include "cublas_example.h"
const int array_size = std::numeric_limits<int>::max() / 2;//-1太容易出现问题了，例如循环溢出

void check_result(int* device_num, int* device_result, int* cpu_num, int* cpu_result);
int* generate_random_array();
void compare1(int* src);
void compare2(int* src);
void compare3(int* src);

