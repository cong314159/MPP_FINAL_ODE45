/*
	main sequence
	data preparation
	different solver call
	timing of the calculation
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "QCA_constants.cuh"
#include "QCA_parameters.cuh"
#include "Support.cuh"
#include "naiveSolver.cuh"
#include "parallelSolver1.cuh"
#include "kernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main(int argc, char* argv[])
{

	clock_t timerStart;
	clock_t timerStop;
	double timer;
	// make a timer

	// data preparation ========================================

	// electronic subsystem Hamiltonian

	matrix Hclk = allocateMatrix(nPosition, nPosition);
	matrixInit(Hclk);
	Hclk.elements[1 * nPosition + 1] = make_cuDoubleComplex(-a * eClk * 0.5, 0);

	matrix Hkin = allocateMatrix(nPosition, nPosition);
	matrixInit(Hkin);
	Hkin.elements[0 * nPosition + 1] = make_cuDoubleComplex(-gamma, 0);
	Hkin.elements[1 * nPosition + 0] = make_cuDoubleComplex(-gamma, 0);
	Hkin.elements[1 * nPosition + 2] = make_cuDoubleComplex(-gamma, 0);
	Hkin.elements[2 * nPosition + 1] = make_cuDoubleComplex(-gamma, 0);

	matrix Hnei = allocateMatrix(nPosition, nPosition); // calculate the coulombic potential
	matrixInit(Hnei);
	double q0_drv = aDriver * qe * (1 - pDriver) / 2;
	double q1_drv = aDriver * qe * (1 + pDriver) / 2;
	double qn_drv = -aDriver * qe;

	double e1, e0, en;
	e1 = q0_drv / (4 * Pi * sqrt(2) * a * 1e-9 * epsilon0) +
		q1_drv / (4 * Pi * sqrt(1) * a * 1e-9 * epsilon0) +
		qn_drv / (4 * Pi * sqrt(2) * a * 1e-9 * epsilon0) -
		q0_drv / (4 * Pi * sqrt(2) * a * 1e-9 * epsilon0) -
		q1_drv / (4 * Pi * sqrt(2) * a * 1e-9 * epsilon0) -
		qn_drv / (4 * Pi * sqrt(1) * a * 1e-9 * epsilon0);
	e0 = q0_drv / (4 * Pi * sqrt(1) * a * 1e-9 * epsilon0) +
		q1_drv / (4 * Pi * sqrt(2) * a * 1e-9 * epsilon0) +
		qn_drv / (4 * Pi * sqrt(2) * a * 1e-9 * epsilon0) -
		q0_drv / (4 * Pi * sqrt(2) * a * 1e-9 * epsilon0) -
		q1_drv / (4 * Pi * sqrt(2) * a * 1e-9 * epsilon0) -
		qn_drv / (4 * Pi * sqrt(1) * a * 1e-9 * epsilon0);
	en = 0;
	Hnei.elements[0 * nPosition + 0] = make_cuDoubleComplex(e0, 0);
	Hnei.elements[1 * nPosition + 1] = make_cuDoubleComplex(en, 0);
	Hnei.elements[2 * nPosition + 2] = make_cuDoubleComplex(e1, 0);

	matrix He = allocateMatrix(nPosition, nPosition);
	matrixInit(He);
	matrixAddInPlace(He, Hclk);
	matrixAddInPlace(He, Hkin);
	matrixAddInPlace(He, Hnei);

	// vibrational hamiltonian
	double m = amu2eV * mAmu / cNms / cNms;
	double f = fInvCms * cCms;
	double omega = 2 * Pi * f;

	cuDoubleComplex * a_data;
	a_data = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * nEnergyState);
	for (int id1 = 0; id1 < nEnergyState; id1++)
	{
		a_data[id1] = make_cuDoubleComplex(sqrt(id1 + 1.0), 0);
	}

	matrix a = diagMatrix(a_data, nEnergyState, 1);

	matrix ad = transpose(a);

	double hbw = hbarEv * omega;

	matrix Hv = matrixScale(matrixAdd(matrixMul(ad, a), matrixScale(eye(nEnergyState), make_cuDoubleComplex(0.5, 0))), make_cuDoubleComplex(hbw, 0));

	cuDoubleComplex g_ev = make_cuDoubleComplex(sqrt(2 * m * omega * omega * lambda), 0);

	matrix X = matrixScale(matrixAdd(ad, a), make_cuDoubleComplex(sqrt(hbarEv / 2 / m / omega), 0));

	matrix sigz = allocateMatrix(nPosition, nPosition);
	matrixInit(sigz);
	sigz.elements[0 * sigz.width + 0] = make_cuDoubleComplex(-1.0, 0.0);
	sigz.elements[2 * sigz.width + 2] = make_cuDoubleComplex(1.0, 0.0);

	matrix Hev = matrixScale(kron(sigz, X), cuCmul(g_ev, make_cuDoubleComplex(0.5,0)));

	matrix He_component = kron(He, eye(nEnergyState));
	matrix Hv_component = kron(eye(nPosition), Hv);
	matrix Hev_component = Hev;

	//matrix H = matrixAdd(matrixAdd(He_component, Hv_component), Hev_component);

	matrix H = allocateMatrix(nPosition * nEnergyState, nPosition * nEnergyState);
	matrixInit(H);
	matrixAddInPlace(H, He_component);
	matrixAddInPlace(H, Hv_component);
	matrixAddInPlace(H, Hev_component);

	double tau = 1.0 / f;
	double T_lindblad = tau / 4;

	matrix lindblad_1 = kron(eye(nPosition), matrixScale(a, make_cuDoubleComplex(sqrt(1.0 / T_lindblad), 0)));
	matrix lindblad_2 = kron(eye(nPosition), matrixScale(ad, make_cuDoubleComplex(exp(-hbw / 2 / kbEv / temp) * sqrt(1 / T_lindblad), 0)));

	lindbladOperators lindblad;
	lindblad.lindblad_1_m = lindblad_1;
	lindblad.lindblad_2_m = lindblad_2;

	// set butcher table for dormand-prince RK45 method
	cuDoubleComplex *C, *B1, *B2;
	cuDoubleComplex *A1, *A2, *A3, *A4, *A5, *A6;

	C = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * 7);
	B1 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * 7);
	B2 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * 7);

	A1 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * 1);
	A2 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * 2);
	A3 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * 3);
	A4 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * 4);
	A5 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * 5);
	A6 = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * 6);

	C[0] = make_cuDoubleComplex(0.0, 0.0);
	C[1] = make_cuDoubleComplex(1.0 / 5.0, 0.0);
	C[2] = make_cuDoubleComplex(3.0 / 10.0, 0.0);
	C[3] = make_cuDoubleComplex(4.0 / 5.0, 0.0);
	C[4] = make_cuDoubleComplex(8.0 / 9.0, 0.0);
	C[5] = make_cuDoubleComplex(1.0, 0.0);
	C[6] = make_cuDoubleComplex(1.0, 0.0);

	B1[0] = make_cuDoubleComplex(35.0 / 384.0, 0.0);
	B1[1] = make_cuDoubleComplex(0.0, 0.0);
	B1[2] = make_cuDoubleComplex(500.0 / 1113.0, 0);
	B1[3] = make_cuDoubleComplex(125.0 / 192.0, 0.0);
	B1[4] = make_cuDoubleComplex(-2187.0 / 6784.0, 0.0);
	B1[5] = make_cuDoubleComplex(11.0 / 84.0, 0.0);
	B1[6] = make_cuDoubleComplex(0.0, 0.0);

	B2[0] = make_cuDoubleComplex(5179.0 / 57600.0, 0.0);
	B2[1] = make_cuDoubleComplex(0.0, 0.0);
	B2[2] = make_cuDoubleComplex(7571.0 / 16695.0, 0.0);
	B2[3] = make_cuDoubleComplex(393.0 / 640.0, 0.0);
	B2[4] = make_cuDoubleComplex(-92097.0 / 339200.0, 0.0);
	B2[5] = make_cuDoubleComplex(187.0 / 2100.0, 0.0);
	B2[6] = make_cuDoubleComplex(1.0 / 40.0, 0.0);

	A1[0] = make_cuDoubleComplex(1.0 / 5.0, 0.0);

	A2[0] = make_cuDoubleComplex(3.0 / 40.0, 0.0);
	A2[1] = make_cuDoubleComplex(9.0 / 40.0, 0.0);

	A3[0] = make_cuDoubleComplex(44.0 / 45.0, 0.0);
	A3[1] = make_cuDoubleComplex(-56.0 / 15.0, 0.0);
	A3[2] = make_cuDoubleComplex(32.0 / 9.0, 0.0);

	A4[0] = make_cuDoubleComplex(19372.0 / 6561.0, 0.0);
	A4[1] = make_cuDoubleComplex(-25360.0 / 2187.0, 0.0);
	A4[2] = make_cuDoubleComplex(64448.0 / 6561.0, 0.0);
	A4[3] = make_cuDoubleComplex(-212.0 / 729.0, 0.0);

	A5[0] = make_cuDoubleComplex(9017.0 / 3168.0, 0.0);
	A5[1] = make_cuDoubleComplex(-355.0 / 33.0, 0.0);
	A5[2] = make_cuDoubleComplex(46732.0 / 5247.0, 0.0);
	A5[3] = make_cuDoubleComplex(49.0 / 176.0, 0.0);
	A5[4] = make_cuDoubleComplex(-5103.0 / 18656.0, 0.0);

	A6[0] = make_cuDoubleComplex(35.0 / 384.0, 0.0);
	A6[1] = make_cuDoubleComplex(0.0, 0.0);
	A6[2] = make_cuDoubleComplex(500.0 / 1113.0, 0.0);
	A6[3] = make_cuDoubleComplex(125.0 / 192.0, 0.0);
	A6[4] = make_cuDoubleComplex(-2187.0 / 6784.0, 0.0);
	A6[5] = make_cuDoubleComplex(11.0 / 84.0, 0.0);

	butcher butcher_DP45;
	butcher_DP45.C_m = C;
	butcher_DP45.B1_m = B1;
	butcher_DP45.B2_m = B2;
	butcher_DP45.A1_m = A1;
	butcher_DP45.A2_m = A2;
	butcher_DP45.A3_m = A3;
	butcher_DP45.A4_m = A4;
	butcher_DP45.A5_m = A5;
	butcher_DP45.A6_m = A6;

	matrix rho_e_Init = allocateMatrix(nPosition, nPosition);
	matrixInit(rho_e_Init);
	rho_e_Init.elements[1 * rho_e_Init.width + 1] = make_cuDoubleComplex(1, 0);
	matrix rho_v_Init = allocateMatrix(nEnergyState, nEnergyState);
	matrixInit(rho_v_Init);
	rho_v_Init.elements[0 * rho_v_Init.width + 0] = make_cuDoubleComplex(1, 0);
	matrix rho = kron(rho_e_Init, rho_v_Init);

	// read rho from a text file from MATLAB calculation
	std::string line;
	std::ifstream rho_eq_file("rho_eq_64.txt");
	if (rho_eq_file.is_open())
	{
		int idx = 0;
		while (getline(rho_eq_file, line))
		{
			char * str = new char[line.length() + 1];
			strcpy(str, line.c_str());

			char * token = strtok(str, "	");

			while (token != NULL)
			{
				std::string tokenString = token;
				rho.elements[idx++] = make_cuDoubleComplex(stod(tokenString), 0);
				token = strtok(NULL, "	");
			}
			delete[] str;
		}
		rho_eq_file.close();
	}

	matrix rhs_init = rhs(rho, H, lindblad);

	// matrixPrint(rhs_init);

	double * time;
	double timeCurrent = 0;
	int timeStep = 0;
	int maxTimeStep = 1;
	double step = 0.01 * tau;
	double maxError = 1E-6;
	double stepUpFactor = 1.1;
	double stepDoFactor = 0.8;

	time = (double *)malloc(sizeof(double) * maxTimeStep);

	//while (timeStep < maxTimeStep)
	//{
	//	std::cout << timeStep << std::endl;
	//	matrix rhoPlus1 = allocateMatrix(rhoSize, rhoSize);
	//	matrix rhoPlus1_ec = allocateMatrix(rhoSize, rhoSize);
	//	matrixInit(rhoPlus1);
	//	matrixInit(rhoPlus1_ec);

	//	stepCalculation(rho, rhoPlus1, rhoPlus1_ec, butcher_DP45, step, lindblad, H);

	//	double error = costFunction(rhoPlus1, rhoPlus1_ec);
	//	std::cout << error << std::endl;
	//	
	//	if (error > maxError)
	//	{
	//		// adjust step size
	//		step *= stepDoFactor;
	//	}
	//	else
	//	{
	//		// output current time and update timeStep
	//		timeCurrent += step;
	//		rho = rhoPlus1;
	//		time[timeStep] = timeCurrent;
	//		timeStep += 1;
	//		step *= stepUpFactor;
	//	}

	//	// matrixPrint(rho);
	//}
	
	// sequential code
	timerStart = clock();
	matrix rhoPlus1 = allocateMatrix(rhoSize, rhoSize);
	matrix rhoPlus1_ec = allocateMatrix(rhoSize, rhoSize);
	matrixInit(rhoPlus1);
	matrixInit(rhoPlus1_ec);
	stepCalculation(rho, rhoPlus1, rhoPlus1_ec, butcher_DP45, step, lindblad, H);
	timerStop = clock();
	timer = (double)(timerStop - timerStart) / CLOCKS_PER_SEC;
	printf("time elapsed in this one step is: %f \n", timer);

	// cuda code

	stepCalculation_p_v1(rho, butcher_DP45, step, lindblad, H);

	return 0;
}