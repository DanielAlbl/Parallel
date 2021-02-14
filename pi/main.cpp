/*
 * Daniel Albl
 * CSC410 2020
 * Program 1
 *
 * This program calculates pi using Monte Carlo approximation
 */

#include <iomanip>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <random>

using namespace std;

const int THREADS = 8;

bool inCircle(double x, double y) {
	// true if distance from the bottom corner is < 1
	return sqrt(x*x + y*y) < 1.0;
}

double getRand() {
	// effectively creates one generator and distribution for each thread
	static thread_local mt19937 gen(time(NULL) + omp_get_thread_num());
    static thread_local uniform_real_distribution<double> dist(0.0,1.0);
	return dist(gen);
}

double MonteCarlo(long long total) {
	long long i, in = 0;

	#pragma omp parallel for reduction(+:in) num_threads(THREADS) schedule(guided)
	for(i = 0; i < total; i++) {
		if(inCircle(getRand(),getRand())) 
			in++;
	}

	return 4.0 * in / total;
}

/*************** MAIN ****************/

int main(int argc, char** argv) {
	double start, end, pi;
	long long itr = stoll(argv[1]);

	start = omp_get_wtime();
	pi = MonteCarlo(itr);
	end = omp_get_wtime();


	cout << "Pi:           " << setprecision(10) << pi << endl; 
	cout << "Time elapsed: " << setprecision(10) << end - start << endl;	
}
