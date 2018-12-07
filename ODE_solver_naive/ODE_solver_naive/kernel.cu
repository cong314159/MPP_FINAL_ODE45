#define BLOCK_SIZE 32

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuComplex.h>
#include <iostream>
#include <time.h>
#include <cublas_v2.h>

#include "QCA_constants.cuh"
#include "QCA_parameters.cuh"
#include "support.cuh"
#include "naiveSolver.cuh"
#include "parallelSolver1.cuh"
#include "kernel.cuh"
#include "util.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void matrixInit_cuda(cuDoubleComplex * mat)
{
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rhoSize && col < rhoSize)
	{
		mat[row * rhoSize + col] = make_cuDoubleComplex(0.0, 0.0);
	}
}

__global__ void matrixScaleAdd_cuda(cuDoubleComplex * A, cuDoubleComplex * B, cuDoubleComplex scaler)
{
	// A += B * scaler
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rhoSize && col < rhoSize)
	{
		A[row * rhoSize + col] = cuCadd(A[row * rhoSize + col], cuCmul(B[row * rhoSize + col], scaler));
	}
}

__global__ void matrixScale_cuda(cuDoubleComplex * A, cuDoubleComplex scaler)
{
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rhoSize && col < rhoSize)
	{
		A[row * rhoSize + col] = cuCmul(A[row * rhoSize + col], scaler);
	}
}

__global__ void matrixSub_cuda(cuDoubleComplex * A, cuDoubleComplex * B)
{
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rhoSize && col < rhoSize)
	{
		A[row * rhoSize + col] = cuCsub(A[row * rhoSize + col], B[row * rhoSize + col]);
	}
}
