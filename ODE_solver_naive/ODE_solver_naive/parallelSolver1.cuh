#ifndef PARALLEL_SOLVER_1
#define PARALLEL_SOLVER_1

void stepCalculation_p_v1(matrix & rho, butcher & butcher_DP45, double step, lindbladOperators & lindblad, matrix & H);

void RHS_cuda(cuDoubleComplex * y, cuDoubleComplex * x, cuDoubleComplex * H, cuDoubleComplex * lindblad1, cuDoubleComplex * lindblad2);

#endif // !PARALLEL_SOLVER_1
