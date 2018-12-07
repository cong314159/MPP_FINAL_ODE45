#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <iostream>

#include "QCA_constants.cuh"
#include "QCA_parameters.cuh"
#include "support.cuh"

matrix allocateMatrix(int height, int width)
{
	matrix mat;
	mat.height = height;
	mat.width = width;
	mat.elements = (cuDoubleComplex *)malloc(height * width * sizeof(cuDoubleComplex));
	matrixInit(mat);

	return mat;
}

void matrixInit(matrix A)
{
	for (int id1 = 0; id1 < A.height; id1++)
	{
		for (int id2 = 0; id2 < A.width; id2++)
		{
			int idx = id1 * A.width + id2;
			A.elements[idx] = make_cuDoubleComplex(0,0);
		}
	}
}

matrix matrixMul(matrix A, matrix B)
{
	matrix mat = allocateMatrix(A.height, B.width);
	int len = A.width;
	for (int id1 = 0; id1 < mat.height; id1++)
	{
		for (int id2 = 0; id2 < mat.width; id2++)
		{
			int idx = id1 * mat.width + id2;
			mat.elements[idx] = make_cuDoubleComplex(0, 0);
			for (int id3 = 0; id3 < len; id3++)
			{
				mat.elements[idx] = cuCadd(mat.elements[idx], cuCmul(A.elements[id1 * A.width + id3], B.elements[id3 * B.width + id2]));
			}
		}
	}
	return mat;
}

matrix matrixMul(matrix A, matrix B, matrix C)
{
	return matrixMul(matrixMul(A, B), C);
}

matrix matrixScale(matrix A, cuDoubleComplex scaler)
{
	matrix B = allocateMatrix(A.height, A.width);
	for (int id1 = 0; id1 < B.height; id1++)
	{
		for (int id2 = 0; id2 < B.width; id2++)
		{
			int idx = id1 * A.width + id2;
			B.elements[idx] = cuCmul(A.elements[idx], scaler);
		}
	}
	return B;
}

matrix matrixScale(matrix A, double scaler)
{
	return matrixScale(A, make_cuDoubleComplex(scaler, 0));
}

void matrixScaleInPlace(matrix A, cuDoubleComplex scaler)
{
	for (int id1 = 0; id1 < A.height; id1++)
	{
		for (int id2 = 0; id2 < A.width; id2++)
		{
			int idx = id1 * A.width + id2;
			A.elements[idx] = cuCmul(A.elements[idx], scaler);
		}
	}
}

void matrixScaleInPlace(matrix A, double scaler)
{
	for (int id1 = 0; id1 < A.height; id1++)
	{
		for (int id2 = 0; id2 < A.width; id2++)
		{
			int idx = id1 * A.width + id2;
			A.elements[idx] = cuCmul(A.elements[idx], make_cuDoubleComplex(scaler,0));
		}
	}
}

matrix diagMatrix(cuDoubleComplex * data, int size, int offset)
{
	matrix mat = allocateMatrix(size, size);
	matrixInit(mat);
	for (int id1 = 0; id1 < size; id1++)
	{
		int idx = id1 * size + id1 + offset;
		if (idx < size * size)
		{
			mat.elements[idx] = data[id1];
		}
	}
	return mat;
}

matrix transpose(matrix A)
{
	matrix mat = allocateMatrix(A.width, A.height);
	for (int id1 = 0; id1 < A.height; id1++)
	{
		for (int id2 = 0; id2 < A.width; id2++)
		{
			mat.elements[id2 * mat.width + id1] = A.elements[id1 * A.width + id2];
		}
	}
	return mat;
}

matrix conj(matrix A)
{
	matrix mat = allocateMatrix(A.width, A.height);
	for (int id1 = 0; id1 < A.height; id1++)
	{
		for (int id2 = 0; id2 < A.width; id2++)
		{
			mat.elements[id2 * mat.width + id1] = cuConj(A.elements[id1 * A.width + id2]);
		}
	}
	return mat;
}

matrix eye(int size)
{
	matrix mat = allocateMatrix(size, size);
	matrixInit(mat);
	for (int id1 = 0; id1 < size; id1++)
	{
		mat.elements[id1 * size + id1] = make_cuDoubleComplex(1, 0);
	}
	return mat;
}

matrix matrixAdd(matrix A, matrix B)
{
	matrix C = allocateMatrix(A.height, A.width);
	for (int id1 = 0; id1 < C.height; id1++)
	{
		for (int id2 = 0; id2 < C.width; id2++)
		{
			int idx = id1 * C.width + id2;
			C.elements[idx] = cuCadd(A.elements[idx], B.elements[idx]);
		}
	}
	return C;
}

void matrixAddInPlace(matrix A, matrix B)
{
	for (int id1 = 0; id1 < A.height; id1++)
	{
		for (int id2 = 0; id2 < A.width; id2++)
		{
			int idx = id1 * A.width + id2;
			A.elements[idx] = cuCadd(A.elements[idx], B.elements[idx]);
		}
	}
}

matrix matrixSub(matrix A, matrix B)
{
	matrix C = allocateMatrix(A.height, A.width);
	matrixInit(C);
	for (int id1 = 0; id1 < C.height; id1++)
	{
		for (int id2 = 0; id2 < C.width; id2++)
		{
			int idx = id1 * C.width + id2;
			C.elements[idx] = cuCsub(A.elements[idx], B.elements[idx]);
		}
	}
	return C;
}

matrix kron(matrix A, matrix B)
{
	matrix C = allocateMatrix(A.height * B.height, A.width * B.width);
	matrixInit(C);
	for (int ida1 = 0; ida1 < A.height; ida1++)
	{
		for (int ida2 = 0; ida2 < A.width; ida2++)
		{
			for (int idb1 = 0; idb1 < B.height; idb1++)
			{
				for (int idb2 = 0; idb2 < B.width; idb2++)
				{
					int idxa, idxb, idx;
					idxa = ida1 * A.width + ida2;
					idxb = idb1 * B.width + idb2;
					idx = (ida1 * B.height + idb1) * C.width + ida2 * B.width + idb2;
					C.elements[idx] = cuCmul(A.elements[idxa], B.elements[idxb]);
				}
			}
		}
	}
	return C;
}

cuDoubleComplex trace(matrix A)
{
	cuDoubleComplex T = make_cuDoubleComplex(0, 0);
	for (int id1 = 0; id1 < A.height; id1++)
	{
		int idx = id1 * A.width + id1;
		T = cuCadd(T, A.elements[idx]);
	}
	return T;
}

matrix commutator(matrix A, matrix B)
{
	return matrixAdd(matrixMul(A, B), matrixScale(matrixMul(B, A), make_cuDoubleComplex(-1.0, 0.0)));
}

void matrixPrint(matrix A)
{
	for (int id1 = 0; id1 < A.height; id1++)
	{
		for (int id2 = 0; id2 < A.width; id2++)
		{
			int idx = id1 * A.width + id2;
			std::cout << cuCreal(A.elements[idx]) << "," << cuCimag(A.elements[idx]) << "  ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void complexPrint(cuDoubleComplex C)
{
	std::cout << cuCreal(C) << "," << cuCimag(C) << std::endl;
}

