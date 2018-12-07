#define BLOCK_SIZE 32

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuComplex.h>
#include <iostream>
#include <time.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>

#include "QCA_constants.cuh"
#include "QCA_parameters.cuh"
#include "support.cuh"
#include "naiveSolver.cuh"
#include "parallelSolver1.cuh"
#include "kernel.cuh"
#include "util.cuh"
#include "cuda_runtime.h"

void stepCalculation_p_v1(matrix & rho, butcher & butcher_DP45, double step, lindbladOperators & lindblad, matrix & H)
{
	cudaError_t cudaReturn;

	cuDoubleComplex *rho_d, *rhoPlus1_d, *rhoPlus1_ec_d, *k1_d, *k2_d, *k3_d, *k4_d, *k5_d, *k6_d, *k7_d;
	cuDoubleComplex *k1_prepare_d, *k2_prepare_d, *k3_prepare_d, *k4_prepare_d, *k5_prepare_d, *k6_prepare_d, *k7_prepare_d;
	cuDoubleComplex *lindblad1_d, *lindblad2_d, *H_d;

	size_t size_all = rhoSize * rhoSize;

	cudaMalloc((void**)&rho_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&rhoPlus1_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&rhoPlus1_ec_d, sizeof(cuDoubleComplex) * size_all);

	cudaMalloc((void**)&k1_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&k2_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&k3_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&k4_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&k5_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&k6_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&k7_d, sizeof(cuDoubleComplex) * size_all);

	cudaMalloc((void**)&k1_prepare_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&k2_prepare_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&k3_prepare_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&k4_prepare_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&k5_prepare_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&k6_prepare_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&k7_prepare_d, sizeof(cuDoubleComplex) * size_all);

	cudaMalloc((void**)&lindblad1_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&lindblad2_d, sizeof(cuDoubleComplex) * size_all);
	cudaMalloc((void**)&H_d, sizeof(cuDoubleComplex) * size_all);

	cudaMemcpy(rho_d, rho.elements, sizeof(cuDoubleComplex) * size_all, cudaMemcpyHostToDevice);
	cudaMemcpy(rhoPlus1_d, rho.elements, sizeof(cuDoubleComplex) * size_all, cudaMemcpyHostToDevice);
	cudaMemcpy(rhoPlus1_ec_d, rho.elements, sizeof(cuDoubleComplex) * size_all, cudaMemcpyHostToDevice);

	cudaMemcpy(lindblad1_d, lindblad.lindblad_1_m.elements, sizeof(cuDoubleComplex) * size_all, cudaMemcpyHostToDevice);
	cudaMemcpy(lindblad2_d, lindblad.lindblad_2_m.elements, sizeof(cuDoubleComplex) * size_all, cudaMemcpyHostToDevice);
	cudaMemcpy(H_d, H.elements, sizeof(cuDoubleComplex) * size_all, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	clock_t timerStart;
	clock_t timerStop;
	double timer;

	timerStart = clock();

	// blockDim, gridDim, ==> launch kernel
	dim3 blockDim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim3((rhoSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (rhoSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

	matrixInit_cuda << <blockDim3, gridDim3 >> > (k1_d);
	matrixInit_cuda << <blockDim3, gridDim3 >> > (k1_prepare_d);
	RHS_cuda(k1_d, rho_d, H_d, lindblad1_d, lindblad2_d);
	matrixScale_cuda << <blockDim3, gridDim3 >> > (k1_d, make_cuDoubleComplex(step, 0));

	matrixInit_cuda << <blockDim3, gridDim3 >> > (k2_d);
	matrixInit_cuda << <blockDim3, gridDim3 >> > (k2_prepare_d);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k2_prepare_d, rho_d, make_cuDoubleComplex(1.0, 0.0));
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k2_prepare_d, k1_d, butcher_DP45.A1_m[0]);
	RHS_cuda(k2_d, k2_prepare_d, H_d, lindblad1_d, lindblad2_d);
	matrixScale_cuda << <blockDim3, gridDim3 >> > (k2_d, make_cuDoubleComplex(step, 0));

	matrixInit_cuda << <blockDim3, gridDim3 >> > (k3_d);
	matrixInit_cuda << <blockDim3, gridDim3 >> > (k3_prepare_d);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k3_prepare_d, rho_d, make_cuDoubleComplex(1.0, 0.0));
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k3_prepare_d, k1_d, butcher_DP45.A2_m[0]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k3_prepare_d, k2_d, butcher_DP45.A2_m[1]);
	RHS_cuda(k3_d, k3_prepare_d, H_d, lindblad1_d, lindblad2_d);
	matrixScale_cuda << <blockDim3, gridDim3 >> > (k3_d, make_cuDoubleComplex(step, 0));

	matrixInit_cuda << <blockDim3, gridDim3 >> > (k4_d);
	matrixInit_cuda << <blockDim3, gridDim3 >> > (k4_prepare_d);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k4_prepare_d, rho_d, make_cuDoubleComplex(1.0, 0.0));
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k4_prepare_d, k1_d, butcher_DP45.A3_m[0]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k4_prepare_d, k2_d, butcher_DP45.A3_m[1]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k4_prepare_d, k3_d, butcher_DP45.A3_m[2]);
	RHS_cuda(k4_d, k4_prepare_d, H_d, lindblad1_d, lindblad2_d);
	matrixScale_cuda << <blockDim3, gridDim3 >> > (k4_d, make_cuDoubleComplex(step, 0));

	matrixInit_cuda << <blockDim3, gridDim3 >> > (k5_d);
	matrixInit_cuda << <blockDim3, gridDim3 >> > (k5_prepare_d);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k5_prepare_d, rho_d, make_cuDoubleComplex(1.0, 0.0));
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k5_prepare_d, k1_d, butcher_DP45.A4_m[0]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k5_prepare_d, k2_d, butcher_DP45.A4_m[1]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k5_prepare_d, k3_d, butcher_DP45.A4_m[2]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k5_prepare_d, k4_d, butcher_DP45.A4_m[3]);
	RHS_cuda(k5_d, k5_prepare_d, H_d, lindblad1_d, lindblad2_d);
	matrixScale_cuda << <blockDim3, gridDim3 >> > (k5_d, make_cuDoubleComplex(step, 0));

	matrixInit_cuda << <blockDim3, gridDim3 >> > (k6_d);
	matrixInit_cuda << <blockDim3, gridDim3 >> > (k6_prepare_d);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k6_prepare_d, rho_d, make_cuDoubleComplex(1.0, 0.0));
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k6_prepare_d, k1_d, butcher_DP45.A5_m[0]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k6_prepare_d, k2_d, butcher_DP45.A5_m[1]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k6_prepare_d, k3_d, butcher_DP45.A5_m[2]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k6_prepare_d, k4_d, butcher_DP45.A5_m[3]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k6_prepare_d, k5_d, butcher_DP45.A5_m[4]);
	RHS_cuda(k6_d, k6_prepare_d, H_d, lindblad1_d, lindblad2_d);
	matrixScale_cuda << <blockDim3, gridDim3 >> > (k6_d, make_cuDoubleComplex(step, 0));

	matrixInit_cuda << <blockDim3, gridDim3 >> > (k7_d);
	matrixInit_cuda << <blockDim3, gridDim3 >> > (k7_prepare_d);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k7_prepare_d, rho_d, make_cuDoubleComplex(1.0, 0.0));
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k7_prepare_d, k1_d, butcher_DP45.A6_m[0]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k7_prepare_d, k2_d, butcher_DP45.A6_m[1]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k7_prepare_d, k3_d, butcher_DP45.A6_m[2]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k7_prepare_d, k4_d, butcher_DP45.A6_m[3]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k7_prepare_d, k5_d, butcher_DP45.A6_m[4]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (k7_prepare_d, k6_d, butcher_DP45.A6_m[5]);
	RHS_cuda(k7_d, k7_prepare_d, H_d, lindblad1_d, lindblad2_d);
	matrixScale_cuda << <blockDim3, gridDim3 >> > (k7_d, make_cuDoubleComplex(step, 0));

	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_d, k1_d, butcher_DP45.B1_m[0]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_d, k2_d, butcher_DP45.B1_m[1]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_d, k3_d, butcher_DP45.B1_m[2]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_d, k4_d, butcher_DP45.B1_m[3]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_d, k5_d, butcher_DP45.B1_m[4]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_d, k6_d, butcher_DP45.B1_m[5]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_d, k7_d, butcher_DP45.B1_m[6]);

	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_ec_d, k1_d, butcher_DP45.B2_m[0]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_ec_d, k2_d, butcher_DP45.B2_m[1]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_ec_d, k3_d, butcher_DP45.B2_m[2]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_ec_d, k4_d, butcher_DP45.B2_m[3]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_ec_d, k5_d, butcher_DP45.B2_m[4]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_ec_d, k6_d, butcher_DP45.B2_m[5]);
	matrixScaleAdd_cuda << <blockDim3, gridDim3 >> > (rhoPlus1_ec_d, k7_d, butcher_DP45.B2_m[6]);

	cudaDeviceSynchronize();

	// free up memory
	cudaFree(k1_d);
	cudaFree(k2_d);
	cudaFree(k3_d);
	cudaFree(k4_d);
	cudaFree(k5_d);
	cudaFree(k6_d);
	cudaFree(k7_d);

	cudaFree(rho_d);
	cudaFree(rhoPlus1_d);
	cudaFree(rhoPlus1_ec_d);

	cudaFree(k1_prepare_d);
	cudaFree(k2_prepare_d);
	cudaFree(k3_prepare_d);
	cudaFree(k4_prepare_d);
	cudaFree(k5_prepare_d);
	cudaFree(k6_prepare_d);
	cudaFree(k7_prepare_d);

	cudaFree(H_d);
	cudaFree(lindblad1_d);
	cudaFree(lindblad2_d);

	std::cout << "step calculation finished" << std::endl;

	timerStop = clock();
	timer = (double)(timerStop - timerStart) / CLOCKS_PER_SEC;
	printf("time elapsed in this one step is: %f \n", timer);
}

void RHS_cuda(cuDoubleComplex * drhodt, cuDoubleComplex * x, cuDoubleComplex * H, cuDoubleComplex * lindblad1, cuDoubleComplex * lindblad2)
{
	dim3 blockDim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim3((rhoSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (rhoSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

	cuDoubleComplex * inter;
	cudaMalloc((void**)&inter, sizeof(cuDoubleComplex) * rhoSize * rhoSize);
	matrixInit_cuda << <gridDim3, blockDim3 >> > (inter);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cuDoubleComplex alpha, beta;

	// lindblad1
	alpha = make_cuDoubleComplex(1, 0);
	beta = make_cuDoubleComplex(1, 0);

	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rhoSize, rhoSize, rhoSize, &alpha, lindblad1, rhoSize, x, rhoSize, &beta, inter, rhoSize);
	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_C, rhoSize, rhoSize, rhoSize, &alpha, inter, rhoSize, lindblad1, rhoSize, &beta, drhodt, rhoSize);

	matrixInit_cuda << <gridDim3, blockDim3 >> > (inter);

	cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, rhoSize, rhoSize, rhoSize, &alpha, lindblad1, rhoSize, lindblad1, rhoSize, &beta, inter, rhoSize);

	alpha = make_cuDoubleComplex(-0.5, 0.0);
	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rhoSize, rhoSize, rhoSize, &alpha, inter, rhoSize, x, rhoSize, &beta, drhodt, rhoSize);
	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rhoSize, rhoSize, rhoSize, &alpha, x, rhoSize, inter, rhoSize, &beta, drhodt, rhoSize);

	// lindblad2
	alpha = make_cuDoubleComplex(1, 0);
	beta = make_cuDoubleComplex(1, 0);

	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rhoSize, rhoSize, rhoSize, &alpha, lindblad2, rhoSize, x, rhoSize, &beta, inter, rhoSize);
	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_C, rhoSize, rhoSize, rhoSize, &alpha, inter, rhoSize, lindblad2, rhoSize, &beta, drhodt, rhoSize);

	matrixInit_cuda << <gridDim3, blockDim3 >> > (inter);

	cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, rhoSize, rhoSize, rhoSize, &alpha, lindblad2, rhoSize, lindblad2, rhoSize, &beta, inter, rhoSize);

	alpha = make_cuDoubleComplex(-0.5, 0.0);
	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rhoSize, rhoSize, rhoSize, &alpha, inter, rhoSize, x, rhoSize, &beta, drhodt, rhoSize);
	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rhoSize, rhoSize, rhoSize, &alpha, x, rhoSize, inter, rhoSize, &beta, drhodt, rhoSize);

	cudaFree(inter);

	// commutator
	alpha = make_cuDoubleComplex(0, -1.0 / hbarEv);
	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rhoSize, rhoSize, rhoSize, &alpha, H, rhoSize, x, rhoSize, &beta, drhodt, rhoSize);

	alpha = make_cuDoubleComplex(0, 1.0 / hbarEv);
	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rhoSize, rhoSize, rhoSize, &alpha, x, rhoSize, H, rhoSize, &beta, drhodt, rhoSize);

	cublasDestroy(handle);
	std::cout << "RHS calculation finished" << std::endl;
}
