#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <iostream>

#include "QCA_constants.cuh"
#include "QCA_parameters.cuh"
#include "support.cuh"
#include "naiveSolver.cuh"

matrix rhs(matrix & rho, matrix & h, lindbladOperators & lindblad)
{
	matrix Lindbladian = allocateMatrix(rho.height, rho.width);
	matrixInit(Lindbladian);

	matrixAddInPlace(Lindbladian, matrixAdd(matrixMul(lindblad.lindblad_1_m, rho, conj(lindblad.lindblad_1_m)), matrixScale(matrixAdd(matrixMul(conj(lindblad.lindblad_1_m), lindblad.lindblad_1_m, rho), matrixMul(rho, conj(lindblad.lindblad_1_m), lindblad.lindblad_1_m)), make_cuDoubleComplex(-0.5, 0))));

	matrixAddInPlace(Lindbladian, matrixAdd(matrixMul(lindblad.lindblad_2_m, rho, conj(lindblad.lindblad_2_m)), matrixScale(matrixAdd(matrixMul(conj(lindblad.lindblad_2_m), lindblad.lindblad_2_m, rho), matrixMul(rho, conj(lindblad.lindblad_2_m), lindblad.lindblad_2_m)), make_cuDoubleComplex(-0.5, 0))));

	matrix drhodt = matrixAdd(matrixScale(commutator(h, rho), make_cuDoubleComplex(0, -1.0 / hbarEv)), Lindbladian);
	// matrix drhodt = matrixScale(commutator(h, rho), make_cuDoubleComplex(0, -1.0 / hbarEv));

	return drhodt;
}

void stepCalculation(matrix & rho, matrix & rhoPlus1, matrix & rhoPlus1_ec, butcher & butcher_DP45, double step, lindbladOperators & lindblad, matrix & H)
{
	matrix k1 = rhs(rho, H, lindblad);
	matrixScaleInPlace(k1, step);

	matrix ak2 = allocateMatrix(rhoSize, rhoSize);
	matrixInit(ak2);
	matrixAddInPlace(ak2, matrixScale(k1, butcher_DP45.A1_m[0]));
	matrix k2 = matrixScale(rhs(matrixAdd(rho, ak2), H, lindblad), step);

	matrix ak3 = allocateMatrix(rhoSize, rhoSize);
	matrixInit(ak3);
	matrixAddInPlace(ak3, matrixScale(k1, butcher_DP45.A2_m[0]));
	matrixAddInPlace(ak3, matrixScale(k2, butcher_DP45.A2_m[1]));
	matrix k3 = matrixScale(rhs(matrixAdd(rho, ak3), H, lindblad), step);

	matrix ak4 = allocateMatrix(rhoSize, rhoSize);
	matrixInit(ak4);
	matrixAddInPlace(ak4, matrixScale(k1, butcher_DP45.A3_m[0]));
	matrixAddInPlace(ak4, matrixScale(k2, butcher_DP45.A3_m[1]));
	matrixAddInPlace(ak4, matrixScale(k3, butcher_DP45.A3_m[2]));
	matrix k4 = matrixScale(rhs(matrixAdd(rho, ak4), H, lindblad), step);

	matrix ak5 = allocateMatrix(rhoSize, rhoSize);
	matrixInit(ak5);
	matrixAddInPlace(ak5, matrixScale(k1, butcher_DP45.A4_m[0]));
	matrixAddInPlace(ak5, matrixScale(k2, butcher_DP45.A4_m[1]));
	matrixAddInPlace(ak5, matrixScale(k3, butcher_DP45.A4_m[2]));
	matrixAddInPlace(ak5, matrixScale(k4, butcher_DP45.A4_m[3]));
	matrix k5 = matrixScale(rhs(matrixAdd(rho, ak5), H, lindblad), step);

	//std::cout << cuCreal(butcher_DP45.A5_m[0]) << "," << cuCimag(butcher_DP45.A5_m[0]) << std::endl;
	//std::cout << cuCreal(butcher_DP45.A5_m[1]) << "," << cuCimag(butcher_DP45.A5_m[1]) << std::endl;
	//std::cout << cuCreal(butcher_DP45.A5_m[2]) << "," << cuCimag(butcher_DP45.A5_m[2]) << std::endl;
	//std::cout << cuCreal(butcher_DP45.A5_m[3]) << "," << cuCimag(butcher_DP45.A5_m[3]) << std::endl;
	//std::cout << cuCreal(butcher_DP45.A5_m[4]) << "," << cuCimag(butcher_DP45.A5_m[4]) << std::endl;

	matrix ak6 = allocateMatrix(rhoSize, rhoSize);
	//matrixInit(ak6);
	matrixAddInPlace(ak6, matrixScale(k1, butcher_DP45.A5_m[0]));
	matrixAddInPlace(ak6, matrixScale(k2, butcher_DP45.A5_m[1]));
	matrixAddInPlace(ak6, matrixScale(k3, butcher_DP45.A5_m[2]));
	matrixAddInPlace(ak6, matrixScale(k4, butcher_DP45.A5_m[3]));
	matrixAddInPlace(ak6, matrixScale(k5, butcher_DP45.A5_m[4]));
	//matrixPrint(ak6);
	matrix k6 = matrixScale(rhs(matrixAdd(rho, ak6), H, lindblad), step);

	//matrixPrint(k6);

	matrix ak7 = allocateMatrix(rhoSize, rhoSize);
	matrixInit(ak7);
	matrixAddInPlace(ak7, matrixScale(k1, butcher_DP45.A6_m[0]));
	matrixAddInPlace(ak7, matrixScale(k2, butcher_DP45.A6_m[1]));
	matrixAddInPlace(ak7, matrixScale(k3, butcher_DP45.A6_m[2]));
	matrixAddInPlace(ak7, matrixScale(k4, butcher_DP45.A6_m[3]));
	matrixAddInPlace(ak7, matrixScale(k5, butcher_DP45.A6_m[4]));
	matrixAddInPlace(ak7, matrixScale(k6, butcher_DP45.A6_m[5]));
	matrix k7 = matrixScale(rhs(matrixAdd(rho, ak7), H, lindblad), step);

	matrixAddInPlace(rhoPlus1, matrixScale(k1, butcher_DP45.B1_m[0]));
	matrixAddInPlace(rhoPlus1, matrixScale(k2, butcher_DP45.B1_m[1]));
	matrixAddInPlace(rhoPlus1, matrixScale(k3, butcher_DP45.B1_m[2]));
	matrixAddInPlace(rhoPlus1, matrixScale(k4, butcher_DP45.B1_m[3]));
	matrixAddInPlace(rhoPlus1, matrixScale(k5, butcher_DP45.B1_m[4]));
	matrixAddInPlace(rhoPlus1, matrixScale(k6, butcher_DP45.B1_m[5]));
	matrixAddInPlace(rhoPlus1, rho);

	matrixAddInPlace(rhoPlus1_ec, matrixScale(k1, butcher_DP45.B2_m[0]));
	matrixAddInPlace(rhoPlus1_ec, matrixScale(k2, butcher_DP45.B2_m[1]));
	matrixAddInPlace(rhoPlus1_ec, matrixScale(k3, butcher_DP45.B2_m[2]));
	matrixAddInPlace(rhoPlus1_ec, matrixScale(k4, butcher_DP45.B2_m[3]));
	matrixAddInPlace(rhoPlus1_ec, matrixScale(k5, butcher_DP45.B2_m[4]));
	matrixAddInPlace(rhoPlus1_ec, matrixScale(k6, butcher_DP45.B2_m[5]));
	matrixAddInPlace(rhoPlus1_ec, matrixScale(k7, butcher_DP45.B2_m[6]));
	matrixAddInPlace(rhoPlus1_ec, rho);
}


double costFunction(matrix & rhoPlus1, matrix & rhoPlus1_ec)
{
	matrix diff = matrixSub(rhoPlus1_ec, rhoPlus1);
	double error = 0.0;
	for (int id1 = 0; id1 < diff.height; id1 ++)
	{
		for (int id2 = 0; id2 < diff.width; id2++)
		{
			int idx = id1 * diff.width + id2;
			error += cuCabs(diff.elements[idx]);
		}
	}
	double size = rhoSize * rhoSize;
	return error / size;
}