#include "cublas_example.h"
# include<stdio.h>
# include<stdlib.h>
# include<math.h>
# include<cuda_runtime.h>
# include"cublas_v2.h"


static __inline__ void modify(	cublasHandle_t handle, 
								float* m, //data
								int ldm,  //因为是列主序，所以是行数
								int n,	  //列数
								int p,	  //行坐标
								int q,	  //列坐标
								float alpha, // scale1
								float beta) //scale2
{
	cublasSscal(handle, n - q, &alpha, &m[IDX2C(p, q, ldm)], ldm);
	cublasSscal(handle, ldm - p, &beta, &m[IDX2C(p, q, ldm)], 1);

}

int test_cublas() {
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

	int i, j;
	float* devPtrA;

	float* a = 0;

	a = (float*)malloc(M * N * sizeof(*a));// 使用*a而非sizeof(float)进行类型自推断


	if (!a) {
		printf("host memory allocation failed\n");

		return EXIT_FAILURE;
		/*
			EXIT_FAILURE: 宏定义，通常为1
			EXIT_SUCCESS：宏定义，通常为0
		*/
	}

	for (j = 0; j < N; j++) {
		for (i = 0; i < M; i++) {
			a[IDX2C(i, j, M)] = (float)(1);
		}
	}

	cudaStat = cudaMalloc((void**)&devPtrA, M * N * sizeof(*a));

	if (cudaStat != cudaSuccess) {
		printf("Device memory allocation failed\n");
		return EXIT_FAILURE;
	}

	stat = cublasCreate(&handle);

	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS initialization failed\n");
		return EXIT_FAILURE;
	}

	// 传输数据到devPtrA
	stat = cublasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("data download failed\n");
		cudaFree(devPtrA);
		cublasDestroy(handle);
		return EXIT_FAILURE;
	}


	modify(handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);
	stat = cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);

	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("data upload failed");
		cudaFree(devPtrA);
		cublasDestroy(handle);
		return EXIT_FAILURE;
	}

	cudaFree(devPtrA);
	cublasDestroy(handle);
	for (j = 0; j < N; j++) {
		for (i = 0; i < M; i++) {
			printf("%7.0f", a[IDX2C(i, j, M)]);
		}
		printf("\n");
	}
	free(a);
	return EXIT_SUCCESS;







}