/*
 * Daniel Albl
 * CSC 410 2020
 * Exam 1
 *
 * This program implements and tests a parallel version of Floyd's algorithm using OpenMP
 */

#include "global.h"

const int THREADS = 8;

// distance from i to j through k, handling Iinites 
int dist(int i, int j, int k) {
	int tmp = at(i, k) + at(k, j);
	return tmp < 0 ? I : tmp;
}

void floyd() {
	int i,j,k;
#pragma omp parallel num_threads(THREADS) shared(A,N) private(i,j,k)
	for (k = 0; k < N; k++) {
#pragma omp for collapse(2) schedule(guided)
		for (i = 0; i < N; i++) 
			for (j = 0; j < N; j++) 
				at(i, j) = min(at(i, j), dist(i, j, k));
	}
}

bool test() {
	A = new int[100];
	bool passed;

	N = 6;
	memcpy(A, t11, 36 * sizeof(int));
	floyd();
	passed = !memcmp(t12, A, 36 * sizeof(int));

	N = 10;
	memcpy(A, t21, 100 * sizeof(int));
	floyd();
	passed = passed && !memcmp(t22, A, 100 * sizeof(int));

	free();
	
	return passed;
}

int main(int argc, char** argv) {
	N = stoi(argv[1]);
	init();
	
	double start = omp_get_wtime();
	floyd();
	double end = omp_get_wtime();
	
	cout << "Time elapsed: " << end - start << "s\n";
	free();
}
