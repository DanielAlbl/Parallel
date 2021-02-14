/*
 * Daniel Albl
 * CSC 410 2020
 * Exam 1
 *
 * This program implements and tests a parallel version of Floyd's algorithm using CUDA 
 */

#include "global.h"

const int BLOCK_SIZE = 256;
int GRID_SIZE, SIZE;
int* _A;

__global__  
void floyd(int* _A, int N) {
	int tmp, idx = blockIdx.x*blockDim.x + threadIdx.x;
	for(int k = 0; k < N; k++) {
		tmp = _A[N*(idx/N) + k] + _A[N*k + (idx%N)];
		if(tmp < _A[idx] and tmp > 0)
			_A[idx] = tmp;
		__syncthreads();
	}
}

bool test() {
	bool passed;
	A = new int[100];
	cudaMalloc((void**)&_A, 100 * sizeof(int));

	cudaMemcpy(_A, t11, 36 * sizeof(int), cudaMemcpyHostToDevice);
	floyd<<<1, 36>>>(_A, 6);
	cudaMemcpy(A, _A, 36 * sizeof(int), cudaMemcpyDeviceToHost);
	passed = !memcmp(t12, A, 36 * sizeof(int));

	cudaMemcpy(_A, t21, 100 * sizeof(int), cudaMemcpyHostToDevice);
	floyd<<<1, 100>>>(_A, 10);
	cudaMemcpy(A, _A, 100 * sizeof(int), cudaMemcpyDeviceToHost);
	passed = passed and !memcmp(t22, A, 100 * sizeof(int));

	free();
	cudaFree(_A);
	
	return passed;
}

int main(int argc, char** argv) {
	N = stoi(argv[1]);
	init();

	SIZE = N*N*sizeof(int);
	GRID_SIZE = N*N / BLOCK_SIZE + (BLOCK_SIZE % N*N) ? 0 : 1;

	// Since you have to copy to gpu in cuda and not omp
	// I think it's fair to include it in the timing
	double start = omp_get_wtime();
	cudaMalloc((void**)&_A, SIZE);
	cudaMemcpy(_A, A, SIZE, cudaMemcpyHostToDevice);
	floyd<<<GRID_SIZE, BLOCK_SIZE>>>(_A, N);
	double end = omp_get_wtime();

	cout << "Time elapsed: " << end - start << "s\n";
	free();
	cudaFree(_A);
}
