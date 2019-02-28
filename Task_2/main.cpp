#include <iostream>
#include <cstdlib>
#include <complex>
#include <mpi.h>

using namespace std;

typedef complex<double> complexd;

int rank, size, s, n;
unsigned long long m, i;

complexd* read(char *f) {
	
	MPI_File file;
	if (MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_RDONLY, MPI_INFO_NULL, &file)) {
		if (!rank) cout << "Error opening file " << f << endl;
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
	}
	
	if (!rank) MPI_File_read(file, &n, 1, MPI_INT, MPI_STATUS_IGNORE);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if (s > n) {
		if (!rank) cout << "Too many processes for this vector" << endl;
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
	}
	
	m = 1LLU << (n - s);
	MPI_File_seek(file, sizeof(int) + 2*m*rank*sizeof(double), MPI_SEEK_SET);
	
	complexd *A = new complexd[m];
	double d[2];
	for (i = 0; i < m; ++i) {
		MPI_File_read(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
		A[i].real(d[0]);
		A[i].imag(d[1]);
	}
	MPI_File_close(&file);
	return A;
	
}

complexd* f(complexd *A, complexd *P, int k) {
	
	complexd *B = new complexd[m];
	
	if (k > s) {
		
		unsigned long long l = 1LLU << (n - k);
		for (i = 0; i < m; ++i)
			B[i] = ((i & l) == 0) ? P[0]*A[i & ~l] + P[1]*A[i | l] : P[2]*A[i & ~l] + P[3]*A[i | l];
		
	} else {
		
		complexd *A1 = new complexd[m];
		int rank1 = rank ^ (1LLU << (s - k));
		
		if (rank < rank1) {
			MPI_Send(A, m, MPI_DOUBLE_COMPLEX, rank1, 0, MPI_COMM_WORLD);
			MPI_Recv(A1, m, MPI_DOUBLE_COMPLEX, rank1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			for (i = 0; i < m; ++i)
				B[i] = P[0]*A[i] + P[1]*A1[i];
		} else {
			MPI_Recv(A1, m, MPI_DOUBLE_COMPLEX, rank1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(A, m, MPI_DOUBLE_COMPLEX, rank1, 0, MPI_COMM_WORLD);
			for (i = 0; i < m; ++i)
				B[i] = P[2]*A1[i] + P[3]*A[i];
		}
		free(A1);
		
	}
	
	return B;
	
}

void write(char *f, complexd *B) {
	
	MPI_File file;
	if (MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file)) {
		if (!rank) cout << "Error opening file " << f << endl;
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
	}
	
	if (!rank) MPI_File_write(file, &n, 1, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_seek(file, sizeof(int) + 2*m*rank*sizeof(double), MPI_SEEK_SET);
	
	double d[2];
	for (i = 0; i < m; ++i) {
		d[0] = B[i].real();
		d[1] = B[i].imag();
		MPI_File_write(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
	}
	MPI_File_close(&file);
	
}

int main(int argc, char **argv) {
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if (argc < 3) {
		if (!rank) cout << argv[0] << " k in [out]" << endl;
		MPI_Finalize();
		return 0;
	}
	int k = atoi(argv[1]);
	
	s = 0;
	while (!((size >> s) & 1)) ++s;
	if ((size >> s) != 1) {
		if (!rank) cout << "Only for 2^n processes" << endl;
		MPI_Finalize();
		return 0;
	}
	
	double time[2], timeMAX[2];
	
	MPI_Barrier(MPI_COMM_WORLD);
	time[0] = MPI_Wtime();
	complexd *A = read(argv[2]);
	time[0] = MPI_Wtime() - time[0];
	
	complexd P[] = {1/sqrt(2), 1/sqrt(2), 1/sqrt(2), -1/sqrt(2)};
	
	MPI_Barrier(MPI_COMM_WORLD);
	time[1] = MPI_Wtime();
	complexd *B = f(A, P, k);
	time[1] = MPI_Wtime() - time[1];
	free(A);
	
	if (argc > 3) write(argv[3], B);
	free(B);
	
	MPI_Reduce(time, timeMAX, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(time+1, timeMAX+1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (!rank) cout << '~' << n << '\t' << k << '\t' << size << '\t' << timeMAX[0] << '\t' << timeMAX[1] << endl;
	
	MPI_Finalize();
	return 0;
	
}
