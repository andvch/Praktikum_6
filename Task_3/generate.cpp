#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv)
{
	
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	
	if (argc < 3) {
		if (!rank)
			cout << argv[0] << " N file" << endl;
		MPI_Finalize();
		return 0;
	}
	
	int n = atoi(argv[1]);
	unsigned long long m = 1LLU << n, k, i;
	k = m/size;
	
	MPI_File file;
	if (MPI_File_open(comm, argv[2], MPI_MODE_CREATE | MPI_MODE_WRONLY,
		MPI_INFO_NULL, &file))
	{
		if (!rank)
			cout << "Error opening file " << argv[2] << endl;
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
		return 1;
	}
	
	if (!rank)
		MPI_File_write(file, &n, 1, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_seek(file, sizeof(int) + 2*k*rank*sizeof(double), MPI_SEEK_SET);
	
	if (rank == size-1)
		k += m%size;
	k <<= 1;
	
	double sqr = 0, module = 0, *d = new double[k];
	unsigned int seed = time(0) + rank;
	for (i = 0; i < k; ++i) {
		d[i] = (rand_r(&seed) / (float) RAND_MAX) - 0.5f;
		sqr += d[i]*d[i];
	}
	
	MPI_Reduce(&sqr, &module, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
	module = sqrt(module);
	MPI_Bcast(&module, 1, MPI_DOUBLE, 0, comm);
	for (i = 0; i < k; ++i)
		d[i] /= module;
	
	MPI_File_write(file, d, k, MPI_DOUBLE, MPI_STATUS_IGNORE);
	
	MPI_File_close(&file);
	MPI_Finalize();
	return 0;
	
}
