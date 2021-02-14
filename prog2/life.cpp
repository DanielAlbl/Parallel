#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include "subGrid.h"

int P, R, I, J, K, M, N;
int m, n; // dimensions of subgrids
bool** G; // main grid
bool* Z; // row of zero
SubGrid* sg; 
MPI_Request* rq;

void setGlobals(char** argv) {
    I = stoi(argv[1]);   
    J = stoi(argv[2]);   
    K = stoi(argv[3]);   
    M = stoi(argv[4]);   
    N = stoi(argv[5]);   
	
	if(R == 0)
		Z = new bool[N+2];

	rq = new MPI_Request[M];
}

void initGrid() {
	int i, j;
	srand(time(NULL));

    G = new bool*[M];
	for(i = 0; i < M; i++)
		G[i] = new bool[N]();

	for(i = 0; i < I; i++)
		G[i/N][i%N] = true;

	// Fisher-Yates shuffle
	for(i = M*N-1; i > 0; i--) {
		j = rand() % (i+1);
		swap(G[i/N][i%N], G[j/N][j%N]);
	}
}

void distribute() {
    bool** g = new bool*[m];

    // make sub array for p=0
    g[0] = new bool[n]();
    for(int i = 1; i < m; i++) {
        g[i] = new bool[n]();
        memcpy(g[i]+1, G[i-1], N*sizeof(bool));
    }

    // make subGrid class for p=0
    sg = new SubGrid(g, m, n, R, P, -1);

    // send sub array data to middle procs
    for(int i = 1; i < P-1; i++) {
        for(int j = 0; j < m; j++) {
            memcpy(g[j]+1, G[i*(m-2)+j-1], N*sizeof(bool));
            MPI_Isend(g[j], n, MPI_CXX_BOOL, i, j, MPI_COMM_WORLD, rq+j);
        }
		MPI_Waitall(m, rq, MPI_STATUSES_IGNORE);
    }
	
	// m for the last subgrid
	int _m = M - (P-1)*(m-2) + 2; 
    for(int i = 0; i < _m-1; i++) {
		memcpy(g[i]+1, G[M-_m+1+i], N*sizeof(bool));
        MPI_Isend(g[i], n, MPI_CXX_BOOL, P-1, i, MPI_COMM_WORLD, rq+i);
	}
	// pad last subgrid with a row of empty cells
    MPI_Isend(Z, n, MPI_CXX_BOOL, P-1, _m-1, MPI_COMM_WORLD, &rq[_m-1]);
	MPI_Waitall(_m, rq, MPI_STATUSES_IGNORE);

	for(int i = 0; i < m; i++)
		delete g[i];
	delete g;
}

void createWorker() {
	// handle the last process having a weird number of rows
	int _m = R == P-1 ? M-(P-1)*(m-2)+2 : m;
	bool** g = new bool*[_m];

	// receive data from main thread
	for(int i = 0; i < _m; i++) {
		g[i] = new bool[n];
		MPI_Irecv(g[i], n, MPI_CXX_BOOL, 0, i, MPI_COMM_WORLD, rq+i);
	}

	MPI_Waitall(_m, rq, MPI_STATUSES_IGNORE);
	sg = new SubGrid(g, _m, n, R, P, R*(m-2)-1);

	for(int i = 0; i < _m; i++)
		delete g[i]; 
	delete g;
}

void gather() {
	// receive data from all the "sendHome" functions
	for(int i = m-2; i < M; i++)
		MPI_Irecv(G[i], N, MPI_CXX_BOOL, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, &rq[i-m+2]);
	MPI_Waitall(M-m+2, rq, MPI_STATUSES_IGNORE);
}

void print() {
	for(int i = 0; i < M; i++) {
		for(int j = 0; j < N; j++) {
			cout << (G[i][j] ? '#':' ') << ' ';
		}
		cout << endl;
	}
	cout << endl;
} 

void free() {
	if(R == 0) { 
		for(int i = 0; i < M; i++)
			delete G[i];
		delete G; delete Z; 
	}
 	delete sg; delete rq;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &R);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    setGlobals(argv);

	m = int(ceil(double(M) / double(P))) + 2;
    n = N + 2;

    if(R == 0) {
		initGrid();
 		distribute();
	}
	else
		createWorker();       

	for(int j = 0; j < J; j++) {
		if(j % K == 0) {
			if(R == 0) {
				cout << "Iteration " << j << ":\n";
				sg->copyToGrid(G);
				gather();
				print();
			}
			else
				sg->sendHome();
		}
		MPI_Barrier(MPI_COMM_WORLD);
		sg->update();
	}

	free();

    MPI_Finalize();
}
   
