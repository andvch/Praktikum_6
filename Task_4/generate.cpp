#include <cstdlib>
#include <mpi.h>
#include <omp.h>
#include "quantum.h"

#define CHECK_ERRORS if (status != 0) { \
	if (!rank) \
		printf("%s", QU::error_comment(status)); \
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
			printf("%s n file\n", argv[0]);
		MPI_Finalize();
		return 0;
	}
	sscanf(argv[1], "%d", &n);
	QU::complexd *A = QU::gen(n, &status);
	CHECK_ERRORS
	status = QU::write(argv[2], A, n);
	CHECK_ERRORS
	delete [] A;
	MPI_Finalize();
	return 0;
}
