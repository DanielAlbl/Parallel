/*
 * Daniel Albl
 * CSC410 2020
 * Program 1
 *
 * This program calculates all prime numbers less than N using the Sieve of Eratosthenes
 */

#include <cstring>
#include <iostream>
#include <math.h>
#include <omp.h>

using namespace std;

long long N;
const int THREADS = 8;
bool* Prime;

void sieveParallel() {
	long long i, j, k, end;
#pragma omp parallel num_threads(THREADS) shared(N,Prime) private(i,j,k)
	for(i = 2; i < N; i+=i) { 
		end = min(2*i,N);
#pragma omp for schedule(guided)
		for(j = i; j < end; j++) 
			if(Prime[j]) 
				for(k = 2*j; k < N; k+=j)
					Prime[k] = false;
	}
}

void printPrimes() {
	cout << "Primes: ";

	long long cnt = 0;
	for(long long i = 2; i < N; i++) {
		if(Prime[i]) {
			if(cnt % 10 == 0)
				cout << endl << cnt << ": ";
			cout << i << " ";
			cnt++;
		}
	}
	cout << "\n\n";
}

/******************* MAIN *******************/
int main(int argc, char** argv) {
	double start, end;
	
	N = stoi(argv[1]);
	Prime = new bool[N];
	memset(Prime,true,N);

	start = omp_get_wtime();
	sieveParallel();
	end = omp_get_wtime();

	printPrimes();
	cout << "Time elapsed: " << 1000 * (end - start) << " ms" << endl;

	delete[] Prime;

	return 0;
}
