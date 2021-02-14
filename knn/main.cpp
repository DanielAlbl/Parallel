#include <iostream>
#include <cstring>
#include <fstream>
#include <string>
#include <random>
#include <math.h>
#include "merge.h"

struct Iris {
	double iv[4]; // independent variables
	int dv; // dependent variable

	Iris() {}
	Iris(double i[4], int d) {
		memcpy(iv, i, 4 * sizeof(double));
		dv = d;
	}
};

int N = 150, K = 10;
Iris T, *D; // test iris, data array
Idx *I, *Tmp; // array of Idx structs, temp array for mergesort
double Mean[4] = { 0 }, Std[4] = { 0 }; // mean, standard deviation of independent vars

void init() {
	D = new Iris[N];
	I = new Idx[N];
	Tmp = new Idx[N];
}

void free() {
	delete D; delete I; delete Tmp;
}

void loadData() {
	double iv[4];
	char c;
	int dv = 0;
	string str;
	ifstream fin("iris.data");

	init();

	int i = 0;
	while (fin >> iv[0] >> c >> iv[1] >> c) {
		fin >> iv[2] >> c >> iv[3] >> c;
		fin >> str;
		if (str == "Iris-setosa")     dv = 1;
		if (str == "Iris-versicolor") dv = 2;
		if (str == "Iris-virginica")  dv = 3;
		D[i++] = Iris(iv, dv);
	}
}

void norm() {
	int i, j;
	#pragma omp parallel private(j)
	{
		#pragma omp for 
		for(i = 0; i < N; i++)
			for(j = 0; j < 4; j++)
				#pragma omp atomic
				Mean[j] += D[i].iv[j];

		#pragma omp master
		for(j = 0; j < 4; j++)
			Mean[j] /= N;

		#pragma omp barrier
		#pragma omp for 
		for(i = 0; i < N; i++)
			for(j = 0; j < 4; j++) {
				D[i].iv[j] -= Mean[j];
				#pragma omp atomic
				Std[j] += D[i].iv[j] * D[i].iv[j];
			}

		#pragma omp master
		for(j = 0; j < 4; j++)
			Std[j] = sqrt(Std[j] / (N-1));

		#pragma omp barrier
		#pragma omp for 
		for(i = 0; i < N; i++)
			for(j = 0; j < 4; j++) 
				D[i].iv[j] /= Std[j];
	}
}

double dist(double* a, double* b, int n) {
	double sum = 0;
	for (int i = 0; i < n; i++)
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	return sqrt(sum);
}

void makeTest(char** argv) {	
	for(int i = 0; i < 4; i++)
		T.iv[i] = (stod(argv[i+1]) - Mean[i]) / Std[i];
}

int argMax3(int* a) {
	if(a[0] > a[1])
		return a[0] > a[2] ? 0:2;
	return a[1] > a[2] ? 1:2;
}

int mode() {
	int cnt[3] = { 0 };
	for(int i = 0; i < K; i++) 
		cnt[D[I[i].idx].dv - 1]++;
	return argMax3(cnt) + 1;
}

int knn() {
	#pragma omp parallel for num_threads(8)
	for(int i = 0; i < N; i++) 
		I[i].set(i, dist(D[i].iv, T.iv, 4));
	mergesort(I, Tmp, N);
	return mode();
}

void print(int s) {
	cout << "Predicted Species: ";
	switch (s) {
		case 1: cout << "Iris-setosa";      break;
		case 2: cout << "Iris-versicolour"; break;
		case 3: cout << "Iris-virginica";   break;
	}
	cout << endl;
}

double time(char** argv) {
   default_random_engine gen;
   normal_distribution<double> dist(0.0, 1.0);
	
	N = stoi(argv[1]);
	init();

	for(int i = 0; i < N; i++)
		for(int j = 0; j < 4; j++)
			D[i].iv[j] = dist(gen);

	double start = omp_get_wtime();
	knn();
	double end = omp_get_wtime();

	return end - start;
}


int main(int argc, char** argv) {
	if(argc == 2) 
		cout << "Time: " << time(argv) << endl;
	else {
		loadData();
		norm();
		makeTest(argv);	
		print(knn());
	}

	free();
}
