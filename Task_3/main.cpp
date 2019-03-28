#include <cstdlib>
#include <stdio.h>
#include <complex>
#include <mpi.h>
#include <omp.h>

using namespace std;

typedef complex<double> complexd;
typedef unsigned long long uint64;

static int rank, size, log_size, threads;

void get_num_threads(int *n)
{
	char *x = getenv("OMP_NUM_THREADS");
	if (x == NULL) {
		*n = 1;
		return;
	}
	sscanf(x, "%d", n);
}

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
	#pragma omp parallel reduction(+: sqr)
	{
		unsigned int seed = time(0) + rank * threads +
			omp_get_thread_num();
		#pragma omp for schedule(guided)
		for (uint64 i = 0; i < m; ++i) {
			A[i].real((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
			A[i].imag((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
			sqr += abs(A[i] * A[i]);
		}
	}
	MPI_Reduce(&sqr, &module, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (!rank)
		module = sqrt(module);
	MPI_Bcast(&module, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	#pragma omp parallel for schedule(guided)
	for (uint64 i = 0; i < m; ++i) {
		A[i] /= module;
	}
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

void quant(complexd *A, complexd *B, int n, int k, complexd *P, complexd *BUF)
{
	uint64 m = num_of_doubles(n);
	if (k > log_size) {
		uint64 l = 1LLU << (n - k);
		#pragma omp parallel for schedule(guided)
		for (uint64 i = 0; i < m; ++i) {
			B[i] = ((i & l) == 0) ?
				P[0]*A[i & ~l] + P[1]*A[i | l] :
				P[2]*A[i & ~l] + P[3]*A[i | l];
		}
	} else {
		int rank1 = rank ^ (1LLU << (log_size - k));
		MPI_Sendrecv(A, m, MPI_DOUBLE_COMPLEX, rank1, 0,
			BUF, m, MPI_DOUBLE_COMPLEX, rank1, 0,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (rank < rank1) {
			#pragma omp parallel for schedule(guided)
			for (uint64 i = 0; i < m; ++i) {
				B[i] = P[0]*A[i] + P[1]*BUF[i];
			}
		} else {
			#pragma omp parallel for schedule(guided)
			for (uint64 i = 0; i < m; ++i) {
				B[i] = P[2]*BUF[i] + P[3]*A[i];
			}
		}
	}
}

double normal_dis_gen(unsigned int *seed)
{
	double S = 0.;
	for (int i = 0; i<12; ++i) {
		S += (double) rand_r(seed) / RAND_MAX;
	}
	return S-6.;
}

complexd *adam(complexd *A, int n, double e)
{
	uint64 m = num_of_doubles(n);
	complexd *B = new complexd[m], *C = new complexd[m], *T,
		*buf = new complexd[m], P[4];
	#pragma omp parallel for schedule(guided)
	for (uint64 i = 0; i < m; ++i) {
		B[i] = A[i];
	}
	double t;
	unsigned int seed = time(0);
	for (int k = 1; k <= n; ++k) {
		if (!rank) {
			t = normal_dis_gen(&seed);
			P[0] = (cos(e*t) - sin(e*t)) / sqrt(2);
			P[1] = P[2] = (cos(e*t) + sin(e*t)) / sqrt(2);
			P[3] = -P[0];
		}
		MPI_Bcast(P, 4, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
		quant(B, C, n, k, P, buf);
		T = B;
		B = C;
		C = T;
	}
	delete [] C;
	delete [] buf;
	return B;
}

complexd dot(complexd *A, complexd *B, int n)
{
	uint64 m = num_of_doubles(n);
	complexd x(0.0, 0.0), y(0.0, 0.0);
	#pragma omp parallel
	{
		complexd z(0.0, 0.0);
		#pragma omp for schedule(guided)
		for (uint64 i = 0; i < m; ++i) {
			z += conj(A[i]) * B[i];
		}
		#pragma omp critical
		y += z;
	}
	MPI_Reduce(&y, &x, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&x, 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
	return x;
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

int init(int argc, char **argv, int *n, double *e)
{
	int i = MPI_Init(&argc, &argv);
	if (i != MPI_SUCCESS) {
		printf("MPI_Init error\n");
		return i;
	}
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	get_num_threads(&threads);
	for (log_size = 0; !((size >> log_size) & 1); ++log_size) {}
	if ((size >> log_size) != 1) {
		if (!rank)
			printf("Only for 2^n processes\n");
		MPI_Finalize();
		return 1;
	}
	if (argc < 3) {
		if (!rank)
			printf("%s n e [in] [out]\n", argv[0]);
		MPI_Finalize();
		return 1;
	}
	sscanf(argv[1], "%d", n);
	sscanf(argv[2], "%lf", e);
	return 0;
}

int main(int argc, char **argv)
{
	int n;
	double e;
	if (init(argc, argv, &n, &e))
		return 0;
	
	double time[3], timeMAX[3];
	MPI_Barrier(MPI_COMM_WORLD);
	time[0] = MPI_Wtime();
	complexd *A = (argc > 3) ? read(argv[3], &n) : gen(n);
	time[0] = MPI_Wtime() - time[0];
	
	MPI_Barrier(MPI_COMM_WORLD);
	time[1] = MPI_Wtime();
	complexd *B = adam(A, n, e);
	time[1] = MPI_Wtime() - time[1];
	
	complexd *C = adam(A, n, 0.0);
	
	delete [] A;
	
	MPI_Barrier(MPI_COMM_WORLD);
	time[2] = MPI_Wtime();
	double lost = abs(dot(B, C, n));
	time[2] = MPI_Wtime() - time[2];
	
	lost = 1.0 - lost * lost;
	if (lost < 0.0)
		lost = 0.0;
	
	if (argc > 4)
		write(argv[4], B, n);
	delete [] B;
	delete [] C;
	MPI_Reduce(time, timeMAX, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (!rank)
		printf("~%d\t%d\t%d\t%lg\t%lf\t|\t%lf\t%lf\t%lf\n",
			size, threads, n, e, lost, timeMAX[0], timeMAX[1], timeMAX[2]);
	
	MPI_Finalize();
	return 0;
}
