#include <omp.h>
using namespace std;

struct Idx {
	int idx;
	double dist;

	void set(int i, double d) { idx = i; dist = d; }
	
	Idx() {}
	Idx(int i, double d) : idx(i), dist(d) {}
	
	friend bool operator<=(Idx const& l, Idx const& r) {
		return l.dist <= r.dist;
	}
};

void merge(Idx* a, Idx* t, int n) { 
	int i = 0, j = 0, k = n/2;

	while(i < n) {
		if(j == n/2) {
			memcpy(a, t, i*sizeof(Idx));
			return;
		}

		if(k == n) {
			memcpy(a+i, a+j, (n-i)*sizeof(Idx));
			memcpy(a, t, i*sizeof(Idx));
			return;
		}

		t[i++] = a[j] <= a[k] ? a[j++] : a[k++];
	}

	memcpy(a, t, n*sizeof(Idx));
}

void mergesort(Idx* a, Idx* t, int n) {
	if(n < 2) return;
	
	#pragma omp task final(n <= 8)
	mergesort(a, t, n/2);
	
	#pragma omp task final(n <= 8)
	mergesort(a+n/2, t+n/2, n-n/2);
	
	#pragma omp taskwait 	
	merge(a, t, n);
}
