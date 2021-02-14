#include <mpi.h>
using namespace std;

class SubGrid {
    bool** M; // subgrid
    bool** C; // copy 
    int m, n, r, p, o; // rows, cols, rank, processors, offset
	MPI_Request* rq; 

	bool updateSquare(int i, int j) {
		// count alive neighbors
        int aliveN = M[i-1][j-1] + M[i-1][j] + M[i-1][j+1] + M[i][j-1] +
                     M[i][j+1] + M[i+1][j-1] + M[i+1][j] + M[i+1][j+1];

		// conway rules
        if(aliveN < 2 or aliveN > 3) return false;
        if(aliveN == 2 and !M[i][j]) return false;
        
        return true;
    }

    void send(bool* ptr, int dest, MPI_Request* rq) {
        MPI_Isend(ptr, n-2, MPI_CXX_BOOL, dest, 0, MPI_COMM_WORLD, rq);
    }

    void recv(bool* ptr, int dest, MPI_Request* rq) {
	    MPI_Irecv(ptr, n-2, MPI_CXX_BOOL, dest, 0, MPI_COMM_WORLD, rq);
    }

    void exchangeData() {
		int cnt = 0;
		
		// send top and bottom rows to above and below neighboring processes
		if(r != 0)   recv(M[ 0 ]+1, r-1, &rq[cnt++]);
        if(r != p-1) recv(M[m-1]+1, r+1, &rq[cnt++]);
        if(r != 0)   send(M[ 1 ]+1, r-1, &rq[cnt++]);
        if(r != p-1) send(M[m-2]+1, r+1, &rq[cnt++]);

		MPI_Waitall(cnt, rq, MPI_STATUSES_IGNORE);
    }

public:
    SubGrid(bool** _M, int m, int n, int r, int p, int o) : m(m), n(n), r(r), p(p), o(o) {	
        M = new bool*[m];
        C = new bool*[m];
		rq = new MPI_Request[m];
        for(int i = 0; i < m; i++) {
            M[i] = new bool[n];
            C[i] = new bool[n];
            memcpy(M[i], _M[i], n*sizeof(bool));
        }
    }
	
    ~SubGrid() {
        for(int i = 0; i < m; i++) {
            delete M[i]; delete C[i];
        }
        delete M; delete C; delete rq;
    }

    void update() {
		// update inside celss
        for(int i = 1; i < m-1; i++) 
            for(int j = 1; j < n-1; j++)
                C[i][j] = updateSquare(i, j);
		
		// copy over
        for(int i = 1; i < m-1; i++)
            memcpy(M[i]+1, C[i]+1, (n-2)*sizeof(bool));

		// exchange border cells between processes
        exchangeData();
    }

	// send data to main process
    void sendHome() {
        for(int i = 1; i < m-1; i++) 
            MPI_Isend(M[i]+1, n-2, MPI_CXX_BOOL, 0, o+i, MPI_COMM_WORLD, &rq[i-1]);            
        MPI_Waitall(m-2, rq, MPI_STATUSES_IGNORE);
    }

	// for proc 0 only
	void copyToGrid(bool** G) {
		for(int i = 1; i < m-1; i++)
			memcpy(G[i-1], M[i]+1, (n-2)*sizeof(bool));
	}
};
