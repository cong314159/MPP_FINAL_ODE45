#ifndef KERNELS
#define KERNELS

__global__ void matrixInit_cuda(cuDoubleComplex * mat);

__global__ void matrixScaleAdd_cuda(cuDoubleComplex * A, cuDoubleComplex * B, cuDoubleComplex scaler);

__global__ void matrixScale_cuda(cuDoubleComplex * A, cuDoubleComplex scaler);

//__global__ void conj_cuda(cuDoubleComplex * A, cuDoubleComplex * B);

#endif // !KERNELS
