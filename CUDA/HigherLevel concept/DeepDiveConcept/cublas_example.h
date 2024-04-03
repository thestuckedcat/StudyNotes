#pragma once
# include<stdio.h>
# include<stdlib.h>
# include<math.h>
# include<cuda_runtime.h>
# include"cublas_v2.h"


#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*ld) + (i))

static __inline__ void modify(cublasHandle_t handle, float* m, int ldm, int n, int p, int q, float alpha, float beta);
int test_cublas();
