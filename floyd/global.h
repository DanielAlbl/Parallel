/*
 * Daniel Albl
 * CSC 410 2020
 * Exam 1
 *
 * This file contians global variables and helper functions used by both cuda.cu and omp.cpp
 */

#ifndef __GLOBAL__
#define __GLOBAL__

#include <iostream>
#include <random>
#include <time.h>
#include <climits>
#include <cstring>
#include <omp.h>
using namespace std;

// GLOBALS
const int I = INT_MAX;
int N; 
int* A;

// TESTS
int t11[36] = { 0,2,5,I,I,I,I,0,7,1,I,8,I,I,0,4,I,I,I,I,I,0,3,I,I,I,2,I,0,3,I,5,I,2,4,0 };
int t12[36] = { 0,2,5,3,6,9,I,0,6,1,4,7,I,15,0,4,7,10,I,11,5,0,3,6,I,8,2,5,0,3,I,5,6,2,4,0 };
int t21[100] = { 0,I,I,14,I,I,I,I,3,11,I,0,1,13,I,9,I,3,3,8,I,3,0,I,I,I,I,3,14,20,I,I,I,0,I,7,I,I,I,I,I,I,I,I,0,I,17,7,14,I,16,I,I,8,I,0,I,I,I,17,I,I,I,12,I,I,0,I,I,I,7,I,I,I,I,I,I,0,I,18,18,I,I,I,1,I,I,I,0,I,1,12,18,I,I,I,8,I,I,0 };
int t22[100] = { 0,23,24,14,4,21,19,11,3,11,9,0,1,13,4,9,16,3,3,8,10,3,0,16,7,12,19,3,6,11,23,36,37,0,27,7,32,34,26,24,14,37,38,28,0,35,17,7,14,25,16,29,30,8,20,0,25,27,19,17,35,48,49,12,39,19,0,46,38,36,7,30,31,21,11,28,26,0,10,18,15,38,39,29,1,36,18,8,0,26,1,12,13,15,5,21,8,12,4,0 };

// helper function for indexing into a cache friendly array
int& at(int i, int j) { return A[N*i + j]; }

// user input and random values
void init() {
	A = new int[N*N];

	mt19937 g(time(NULL));
	uniform_int_distribution<int> d1(0,3); // 25% chance of an edge
	uniform_int_distribution<int> d2(1, 20); // random edge of wieghts 1-20 inclusive 

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i == j) 
				at(i, j) = 0;
			else 
				at(i, j) = d1(g) ? I : d2(g);
		}
	}
}

void free() { delete A; }

#endif
