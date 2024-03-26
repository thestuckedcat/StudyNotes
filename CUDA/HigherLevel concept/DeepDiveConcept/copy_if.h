#pragma once
# include <cuda.h>
# include <cuda_fp16.h>

__global__ void origin_copy_if(int* res, const int* src, int* nres, int n);

__host__ int cpu_copy_if(int* res, const int* src, int n);

__global__ void blocklevel_copy_if(int* res, const int* src, int* nres, int n);

__global__ void warplevel_copy_if(int* res, const int* src, int* nres, int n);