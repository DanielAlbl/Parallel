#include <iostream>
#include <mpi.h>
using namespace std;

int rnk, n;
char* data;

double blocking() {
	double start = MPI_Wtime();
	if(rnk == 0) {
		MPI_Send(data, n, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
		MPI_Recv(data, n, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else {
		MPI_Recv(data, n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(data, n, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	}
	double end = MPI_Wtime();

	return (end - start) / 2;
}

double nonBlocking() {
	MPI_Request rq[2];

	double start = MPI_Wtime();
	if(rnk == 0) {
		MPI_Irecv(data, n, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &rq[1]);
		MPI_Isend(data, n, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &rq[0]);
		MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
	}
	else {
		MPI_Irecv(data, n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &rq[1]);
		MPI_Isend(data, n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &rq[0]);
		MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
	}
	double end = MPI_Wtime();

	return (end - start) / 2;
}

double sendRecv() {
	double start = MPI_Wtime();
	if(rnk == 0)
		MPI_Sendrecv(data, n, MPI_CHAR, 1, 0, data, n, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
	else
		MPI_Sendrecv(data, n, MPI_CHAR, 0, 0, data, n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
	double end = MPI_Wtime();

 	return (end - start) / 2;
}

int main (int argc, char *argv[]) {
    if(argc != 2) {
        printf("Must provide number of bytes as a command argument\n");
        exit(1);
    }

    n = stoi(argv[1]);
    data = new char[n];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rnk);

    MPI_Barrier(MPI_COMM_WORLD);
 
	double time = sendRecv();

    if(rnk == 0)
        cout << "Time per message: " << 1000 * time << " ms\n";

    MPI_Finalize();

    delete data; 
}
