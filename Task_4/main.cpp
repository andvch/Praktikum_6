#include <cstdlib>
#include <mpi.h>
#include <omp.h>
#include "quantum.h"

#define CHECK_ERRORS if (status != 0) { \
	if (!rank) \
		puts(QU::error_comment(status)); \
	MPI_Finalize(); \
	return status; \
}

int main(int argc, char **argv)
{
	int rank, n, status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (argc < 3) {
		if (!rank)
			printf("%s in out\n", argv[0]);
		MPI_Finalize();
		return 0;
	}

	QU::complexd *A = QU::read(argv[1], &n, &status), *B = nullptr;
	CHECK_ERRORS

	int t, k[2];
	double a;
	if (!rank) {
		printf("Choose transformation\n");
		printf("1\tHadamard\n2\tnHadamard\n3\tR\n4\tNOT\n5\tCNOT\n6\tCR\n");
		if (scanf("%u", &t) != 1)
			return 5;
	}
	MPI_Bcast(&t, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (t == 1 || t == 3 || t == 4) {
		if (!rank) {
			printf("Enter k\n");
			if (scanf("%u", k) != 1)
				return 5;
		}
		MPI_Bcast(k, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}
	if (t == 5 || t == 6) {
		if (!rank) {
			printf("Enter k1 k2\n");
			if (scanf("%u %u", k, k+1) != 2)
				return 5;
		}
		MPI_Bcast(k, 2, MPI_INT, 0, MPI_COMM_WORLD);
	}
	if (t == 3 || t == 6) {
		if (!rank) {
			printf("Enter a\n");
			if (scanf("%lf", &a) != 1)
				return 5;
		}
		MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	switch(t) {
		case 1: B = QU::Hadamard(A, n, k[0], &status); break;
		case 2: B = QU::nHadamard(A, n, &status); break;
		case 3: B = QU::R(A, n, k[0], a, &status); break;
		case 4: B = QU::NOT(A, n, k[0], &status); break;
		case 5: B = QU::CNOT(A, n, k[0], k[1], &status); break;
		case 6: B = QU::CR(A, n, k[0], k[1], a, &status); break;
		default: break;
	}
	CHECK_ERRORS
	status = QU::write(argv[2], B, n);
	CHECK_ERRORS
	delete [] A;
	delete [] B;
	MPI_Finalize();
	return 0;
}
