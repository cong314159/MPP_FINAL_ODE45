#ifndef SUPPORT
#define SUPPORT

struct matrix
{
	int height;
	int width;
	cuDoubleComplex * elements;
};

struct butcher
{
	cuDoubleComplex *C_m, *B1_m, *B2_m;
	cuDoubleComplex *A1_m, *A2_m, *A3_m, *A4_m, *A5_m, *A6_m;
};

struct lindbladOperators
{
	matrix lindblad_1_m;
	matrix lindblad_2_m;
};

matrix allocateMatrix(int height, int width);

void matrixInit(matrix A);

matrix matrixMul(matrix A, matrix B);

matrix matrixMul(matrix A, matrix B, matrix C);

matrix matrixScale(matrix A, cuDoubleComplex scaler);

matrix matrixScale(matrix A, double scaler);

void matrixScaleInPlace(matrix A, cuDoubleComplex scaler);

void matrixScaleInPlace(matrix A, double scaler);

matrix diagMatrix(cuDoubleComplex * data, int size, int offset);

matrix transpose(matrix A);

matrix conj(matrix A);

matrix eye(int size);

matrix matrixAdd(matrix A, matrix B);

void matrixAddInPlace(matrix A, matrix B);

matrix matrixSub(matrix A, matrix B);

matrix kron(matrix A, matrix B);

cuDoubleComplex trace(matrix A);

matrix commutator(matrix A, matrix B);

void matrixPrint(matrix A);

void complexPrint(cuDoubleComplex C);

#endif // !SUPPORT
