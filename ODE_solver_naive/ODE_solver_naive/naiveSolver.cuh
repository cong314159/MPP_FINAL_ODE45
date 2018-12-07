#ifndef NAIVE_SOLVER
#define NAIVE_SOLVER

// #include "support.cuh"

// rhs function
matrix rhs(matrix & rho, matrix & h, lindbladOperators & lindblad);

// find next rho and rho_error_correction
void stepCalculation(matrix & rho, matrix & rhoPlus1, matrix & rhoPlus1_ec, butcher & butcher_DP45, double step, lindbladOperators & lindblad, matrix & H);

double costFunction(matrix & rhoPlus1, matrix & rhoPlus1_ec);

#endif // !NAIVE_SOLVER
