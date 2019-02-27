#include <iostream>
#include <cstdlib>
#include <complex>
#include <mpi.h>

using namespace std;

typedef complex<double> complexd;

int main(int argc, char **argv) {
	
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	
	if (argc < 3) {
		if (!rank) cout << argv[0] << " k in [out]" << endl;
		MPI_Finalize();
		return 0;
	}
	int k = atoi(argv[1]), x = 0;
	
	while (!((size >> x) & 1)) ++x;
	if ((size >> x) != 1) {
		if (!rank) cout << "Only for 2^n processes" << endl;
		MPI_Finalize();
		return 0;
	}
	
	MPI_File in;
	if (MPI_File_open(comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &in)) {
		if (!rank) cout << "Error opening file " << argv[2] << endl;
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
		return 1;
	}
	
	int n;
	if (!rank) MPI_File_read(in, &n, 1, MPI_INT, MPI_STATUS_IGNORE);
	MPI_Bcast(&n, 1, MPI_INT, 0, comm);
	
	unsigned long long m = 1LLU << (n - x), i;
	MPI_File_seek(in, sizeof(int) + 2*m*rank*sizeof(double), MPI_SEEK_SET);
	
	double d[2];
	complexd *A = new complexd[m];
	complexd *B = new complexd[m];
	for (i = 0; i < m; ++i) {
		MPI_File_read(in, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
		A[i].real() = d[0];
		A[i].imag() = d[1];
	}
	MPI_File_close(&in);
	
	
	complexd P[] = {1/sqrt(2), 1/sqrt(2), 1/sqrt(2), -1/sqrt(2)};
	
	if (k > x) {
		
		unsigned long long l = 1LLU << (n - k);
		for (i = 0; i < m; ++i)
			B[i] = ((i & l) == 0) ? P[0]*A[i & ~l] + P[1]*A[i | l] : P[2]*A[i & ~l] + P[3]*A[i | l];
		
	} else {
		
		complexd *A1 = new complexd[m];
		int rank1 = rank ^ (1LLU << (x - k));
		if (rank < rank1) {
			MPI_Send(A, m, MPI_DOUBLE_COMPLEX, rank1, 0, comm);
			MPI_Recv(A1, m, MPI_DOUBLE_COMPLEX, rank1, 0, comm, MPI_STATUS_IGNORE);
			for (i = 0; i < m; ++i)
				B[i] = P[0]*A[i] + P[1]*A1[i];
		} else {
			MPI_Recv(A1, m, MPI_DOUBLE_COMPLEX, rank1, 0, comm, MPI_STATUS_IGNORE);
			MPI_Send(A, m, MPI_DOUBLE_COMPLEX, rank1, 0, comm);
			for (i = 0; i < m; ++i)
				B[i] = P[2]*A1[i] + P[3]*A[i];
		}
		free(A1);
		
	}
	free(A);
	
	if (argc == 3) {
		free(B);
		MPI_Finalize();
		return 0;
	}
	
	MPI_File out;
	if (MPI_File_open(comm, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out)) {
		if (!rank) cout << "Error opening file " << argv[3] << endl;
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
		return 1;
	}
	
	if (!rank) MPI_File_write(out, &n, 1, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_seek(out, sizeof(int) + 2*m*rank*sizeof(double), MPI_SEEK_SET);
	for (i = 0; i < m; ++i) {
		d[0] = A[i].real();
		d[1] = A[i].imag();
		MPI_File_write(out, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
	}
	MPI_File_close(&out);
	free(B);
	
	MPI_Finalize();
	return 0;
	
}
