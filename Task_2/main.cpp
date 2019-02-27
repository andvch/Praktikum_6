#include <iostream>
#include <cstdlib>
#include <complex>
#include <mpi.h>

using namespace std;

typedef complex<double> complexd;

int main(int argc, char **argv) {
	
	int rank, size, k = 0;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	
	if (argc < 3) {
		if (!rank) cout << argv[0] << " k in [out]" << endl;
		MPI_Finalize();
		return 0;
	}
	
	while (!((size >> k) & 1)) ++k;
	if ((size >> k) != 1) {
		if (!rank) cout << "Only for 2^n processes" << endl;
		MPI_Finalize();
		return 0;
	}
	
	MPI_File in, out;
	if (MPI_File_open(comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &in)) {
		if (!rank) cout << "Error opening file " << argv[2] << endl;
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
		return 1;
	}
	
	int n;
	if (!rank) MPI_File_read(in, &n, 1, MPI_INT, MPI_STATUS_IGNORE);
	MPI_Bcast(&n, 1, MPI_INT, 0, comm);
	
	if (k > n) k = n;
	unsigned long long m = 1LLU << (n - k), i;
	k = atoi(argv[1]);
	
	MPI_File_seek(in, sizeof(int) + 2*m*rank*sizeof(double), MPI_SEEK_SET);
	
	double d[2];
	complexd *a = new complexd[m];
	for (i = 0; i < m; ++i) {
		MPI_File_read(in, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
		a[i].real() = d[0];
		a[i].imag() = d[1];
	}
	
//	.....
	
	MPI_File_close(&in);
	MPI_Finalize();
	return 0;
	
}
