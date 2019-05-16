#include <cstdlib>
#include <mpi.h>
#include <omp.h>
#include "../Task_4/quantum.h"

static int status;

#define CHECK_ERRORS if (status != 0) { \
	if (!rank) \
		puts(QU::error_comment(status)); \
	MPI_Finalize(); \
	return status; \
}

int main(int argc, char **argv)
{
	int rank, n;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (argc < 2) {
		if (!rank)
			printf("%s n [in] [out]\n", argv[0]);
		MPI_Finalize();
		return 0;
	}

	sscanf(argv[1], "%d", &n);
	QU::complexd *A = (argc > 2) ? QU::read(argv[2], &n, &status) :
		QU::gen(n, &status), *B = nullptr;
	CHECK_ERRORS

	int i, j;
	double time = MPI_Wtime(), timeMAX;
	for (i = 1; i <= n; ++i) {
		B = QU::Hadamard(A, n, i, &status);
		CHECK_ERRORS
		delete [] A;
		A = B;
		for (j = i + 1; j <= n; ++j) {
			B = QU::CR(A, n, i, j, M_PI / (1 << (j - i)), &status);
			CHECK_ERRORS
			delete [] A;
			A = B;
		}
	}
	time = MPI_Wtime() - time;
	MPI_Reduce(&time, &timeMAX, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (!rank)
		printf("~%d\t%d\t%d\t%lf\n",
			n, QU::details::size, QU::details::threads, timeMAX);

	QU::complexd P0[] = {1, 0, 0, 0};
	QU::complexd P1[] = {0, 0, 1, 0};
	QU::complexd P2[] = {0, 1, 0, 0};
	QU::complexd P3[] = {0, 0, 0, 1};
	QU::complexd *P[] = {P0, P1, P2, P3};
	QU::uint k[2];
	for (i = 0; i < n / 2; ++i) {
		k[0] = i + 1;
		k[1] = n - i;
		B = QU::transformation(A, n, 2, P, k, &status);
		CHECK_ERRORS
		delete [] A;
		A = B;
	}

	if (argc > 3) {
		status = QU::write(argv[3], A, n);
		CHECK_ERRORS
	}
	delete [] A;

	MPI_Finalize();
	return 0;
}
