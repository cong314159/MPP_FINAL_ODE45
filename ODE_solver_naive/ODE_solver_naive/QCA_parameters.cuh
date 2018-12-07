#ifndef PARAMETERS
#define PARAMETERS

const double a = 1.0; // [nm] the distance between '1' and '0' location
const double pDriver = 1.0; // [no unit]
const double aDriver = 0.9; // [no unit]
const double gamma = 0.1; // [eV] tunneling energy
const double temp = 300.0; // [K]
const double fInvCms = 298.0; // [ / cm] oscillator frequency
const double mAmu = 5.58; // [amu]
const double lambda = 0.44; // [eV]
const int nEnergyState = 64; // []
const int nPosition = 3;
const int rhoSize = nEnergyState * nPosition;

const double pTarget0 = 1.0;
const double aTarget0 = 0.9;
const double eClk = 0; // [V / nm]

#endif // !PARAMETERS
