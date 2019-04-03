#include <cstdlib>
#include <stdio.h>
#include <complex>
#include <mpi.h>

using namespace std;

typedef complex<double> complexd;
typedef unsigned long long uint64;

static int rank, size, log_size;

uint64 num_of_doubles(int n)
{
	if (log_size > n) {
		if (!rank)
			printf("Too many processes for this n\n");
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
	}
	return 1LLU << (n - log_size);
}

complexd* gen(int n)
{
	uint64 m = num_of_doubles(n);
	complexd *A = new complexd[m];
	double sqr = 0, module;
	unsigned int seed = time(0) + rank;
	for (uint64 i = 0; i < m; ++i) {
		A[i].real((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
		A[i].imag((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
		sqr += abs(A[i] * A[i]);
	}
	MPI_Reduce(&sqr, &module, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (!rank)
		module = sqrt(module);
	MPI_Bcast(&module, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	for (uint64 i = 0; i < m; ++i)
		A[i] /= module;
	return A;
}

complexd* read(char *f, int *n)
{
	MPI_File file;
	if (MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_RDONLY, MPI_INFO_NULL,
		&file))
	{
		if (!rank)
			printf("Error opening file %s\n", f);
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
	}
	if (!rank)
		MPI_File_read(file, n, 1, MPI_INT, MPI_STATUS_IGNORE);
	MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	uint64 m = num_of_doubles(*n);
	complexd *A = new complexd[m];
	double d[2];
	MPI_File_seek(file, sizeof(int) + 2 * m * rank * sizeof(double),
		MPI_SEEK_SET);
	for (uint64 i = 0; i < m; ++i) {
		MPI_File_read(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
		A[i].real(d[0]);
		A[i].imag(d[1]);
	}
	MPI_File_close(&file);
	return A;
}

complexd *quant(complexd *A, int n, int k, complexd *P)
{
	uint64 m = num_of_doubles(n);
	complexd *B = new complexd[m];
	if (k > log_size) {
		uint64 l = 1LLU << (n - k);
		for (uint64 i = 0; i < m; ++i)
			B[i] = ((i & l) == 0) ?
				P[0]*A[i & ~l] + P[1]*A[i | l] :
				P[2]*A[i & ~l] + P[3]*A[i | l];
	} else {
		complexd *BUF = new complexd[m];
		int rank1 = rank ^ (1LLU << (log_size - k));
		MPI_Sendrecv(A, m, MPI_DOUBLE_COMPLEX, rank1, 0,
			BUF, m, MPI_DOUBLE_COMPLEX, rank1, 0,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (rank < rank1) {
			for (uint64 i = 0; i < m; ++i)
				B[i] = P[0]*A[i] + P[1]*BUF[i];
		} else {
			for (uint64 i = 0; i < m; ++i)
				B[i] = P[2]*BUF[i] + P[3]*A[i];
		}
		delete [] BUF;
	}
	return B;
}

void write(char *f, complexd *B, int n)
{
	MPI_File file;
	if (MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_CREATE | MPI_MODE_WRONLY,
		MPI_INFO_NULL, &file))
	{
		if (!rank)
			printf("Error opening file %s\n", f);
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
	}
	if (!rank)
		MPI_File_write(file, &n, 1, MPI_INT, MPI_STATUS_IGNORE);
	uint64 m = num_of_doubles(n);
	double d[2];
	MPI_File_seek(file, sizeof(int) + 2*m*rank*sizeof(double), MPI_SEEK_SET);
	for (uint64 i = 0; i < m; ++i) {
		d[0] = B[i].real();
		d[1] = B[i].imag();
		MPI_File_write(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
	}
	MPI_File_close(&file);
}

int init(int argc, char **argv, int *n, int *k)
{
	int i = MPI_Init(&argc, &argv);
	if (i != MPI_SUCCESS) {
		printf("MPI_Init error\n");
		return i;
	}
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	for (log_size = 0; !((size >> log_size) & 1); ++log_size) {}
	if ((size >> log_size) != 1) {
		if (!rank)
			printf("Only for 2^n processes\n");
		MPI_Finalize();
		return 1;
	}
	if (argc < 3) {
		if (!rank)
			printf("%s n k [in] [out]\n", argv[0]);
		MPI_Finalize();
		return 1;
	}
	sscanf(argv[1], "%d", n);
	sscanf(argv[2], "%d", k);
	return 0;
}

int main(int argc, char **argv)
{
	int n, k;
	if (init(argc, argv, &n, &k))
		return 0;
	
	double time[2], timeMAX[2];
	MPI_Barrier(MPI_COMM_WORLD);
	time[0] = MPI_Wtime();
	complexd *A = (argc > 3) ? read(argv[3], &n) : gen(n);
	time[0] = MPI_Wtime() - time[0];
	
	complexd P[] = {1/sqrt(2), 1/sqrt(2), 1/sqrt(2), -1/sqrt(2)};
	
	MPI_Barrier(MPI_COMM_WORLD);
	time[1] = MPI_Wtime();
	complexd *B = quant(A, n, k, P);
	time[1] = MPI_Wtime() - time[1];
	
	if (argc > 4)
		write(argv[4], B, n);
	delete [] B;
	MPI_Reduce(time, timeMAX, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (!rank)
		printf("~%d\t%d\t%d\t%lf\t%lf\n",
			n, k, size, timeMAX[0], timeMAX[1]);
	
	MPI_Finalize();
	return 0;
}
