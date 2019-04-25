#ifndef TASK_4_QUANTUM_H_
#define TASK_4_QUANTUM_H_

#include <cstdlib>
#include <ctime>
#include <cmath>
#include <complex>
#include <map>
#include <mpi.h>
#include <omp.h>

namespace QU
{
typedef std::complex<double> complexd;
typedef unsigned int uint;
typedef unsigned long long uint64;

namespace details
{
bool init_flag = false;
int rank, size, log_size, threads;

static inline void get_num_threads(int *n)
{
	*n = 1;
	char *x = getenv("OMP_NUM_THREADS");
	if (x == NULL)
		return;
	sscanf(x, "%d", n);
}

static inline int init(void)
{
	if (init_flag)
		return 0;
	int status;
	MPI_Initialized(&status);
	if (!status)
		return 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	get_num_threads(&threads);
	for (log_size = 0; ((size >> log_size) & 1) == 0; ++log_size)
		{}
	if ((size >> log_size) != 1)
		return 2;
	init_flag = true;
	return 0;
}

static inline uint64 num_of_doubles(int n)
{
	if (log_size > n)
		return 0;
	return 1LLU << (n - log_size);
}

template <typename T>
static inline T* get_masks(uint i, const uint* k, uint len)
{
	uint l = 1 << i;
	T *masks = new T[l];
	for (uint j = 0; j < l; ++j) {
		masks[j] = 0;
		for (uint p = 0; p < i; ++p)
			if (((j >> p) & 1) == 1)
				masks[j] ^= 1LLU << (len - k[i-1 - p]);
	}
	return masks;
}

static inline uint* get_ranks(uint i, const uint* k)
{
	uint p[i], j = 0;
	for (uint t = 0; t < i; ++t)
		if (k[t] <= (uint) log_size)
			p[j++] = k[t];
	uint* ranks = get_masks<uint>(j, p, log_size);

	ranks[0] = 1 << j;
	for (uint t = 1; t < ranks[0]; ++t)
		ranks[t] ^= rank;
	--ranks[0];
	return ranks;
}

static inline int trans(const complexd *A, complexd *B, uint n,
	uint i, complexd **P, const uint *k, complexd *BUF = nullptr)
{
	int status = init();
	if (status != 0)
		return status;
	uint64 m = num_of_doubles(n);
	if (m == 0)
		return 3;

	uint l = 1 << i;
	uint64 *masks = get_masks<uint64>(i, k, n);
	uint *ranks = get_ranks(i, k);

	bool flag = false;
	if ((BUF == nullptr) && (ranks[0] > 0)) {
		BUF = new complexd[m*ranks[0]];
		flag = true;
	}

	std::map<uint, uint> rank2id;
	for (uint j = 0; j < ranks[0]; ++j) {
		MPI_Sendrecv(A, m, MPI_CXX_DOUBLE_COMPLEX, ranks[j+1], 0,
			BUF + m*j, m, MPI_CXX_DOUBLE_COMPLEX, ranks[j+1], 0,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		rank2id[ranks[j+1]] = j;
	}

	#pragma omp parallel
	{
		complexd *vec = new complexd[l];
		uint x = 0, id0;			// P line number, rank id
		uint64 tmp, num, id1;		// element number, local id
		const uint64 tmP0 = rank << (n - log_size),
			tmp0 = ~((~0LLU) << (n - log_size));

		#pragma omp for schedule(guided)
		for (uint64 j = 0; j < m; ++j) {
			tmp = (tmp0 ^ j) & ~masks[l-1];
			for (uint z = 0; z < l; ++z) {
				num = tmp ^ masks[z];
				id0 = num >> (n - log_size);
				id1 = num & tmP0;
				if (id0 == (uint) rank) {
					vec[z] = A[id1];
					if (id1 == j)
						x = z;
				} else {
					vec[z] = BUF[m*rank2id[id0] + id1];
				}
			}
			B[j] = 0;
			for (uint z = 0; z < l; ++z) {
				B[j] += P[x][z]*vec[z];
			}
		}
		delete [] vec;
	}

	if (flag)
		delete [] BUF;
	delete [] masks;
	delete [] ranks;

	return 0;
}
}  // namespace details


static inline complexd* gen(uint n, int *status = nullptr)
{
	int st;
	if (status == nullptr)
		status = &st;

	*status = details::init();
	if (*status != 0)
		return NULL;
	uint64 m = details::num_of_doubles(n);
	if (m == 0) {
		*status = 3;
		return NULL;
	}

	complexd *A = new complexd[m];
	double sqr = 0, module;
	#pragma omp parallel reduction(+: sqr)
	{
		unsigned int seed = time(0) +
			details::rank * details::threads + omp_get_thread_num();
		#pragma omp for schedule(guided)
		for (uint64 i = 0; i < m; ++i) {
			A[i].real(rand_r(&seed) / static_cast<float>(RAND_MAX) - 0.5f);
			A[i].imag(rand_r(&seed) / static_cast<float>(RAND_MAX) - 0.5f);
			sqr += abs(A[i] * A[i]);
		}
	}
	MPI_Reduce(&sqr, &module, 1, MPI_DOUBLE, MPI_SUM, 0,
		MPI_COMM_WORLD);
	if (!details::rank)
		module = sqrt(module);
	MPI_Bcast(&module, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	#pragma omp parallel for schedule(guided)
	for (uint64 i = 0; i < m; ++i) {
		A[i] /= module;
	}
	return A;
}

static inline complexd* read(const char *f, int *n,
	int *status = nullptr)
{
	int st;
	if (status == nullptr)
		status = &st;

	*status = details::init();
	if (*status != 0)
		return NULL;

	MPI_File file;
	if (MPI_File_open(MPI_COMM_WORLD, f,
		MPI_MODE_RDONLY, MPI_INFO_NULL, &file))
	{
		*status = 4;
		return NULL;
	}
	*n = 0;
	if (!details::rank)
		MPI_File_read(file, (unsigned char*) n, 1,
			MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
	MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	uint64 m = details::num_of_doubles(*n);
	if (m == 0) {
		MPI_File_close(&file);
		*status = 3;
		return NULL;
	}
	complexd *A = new complexd[m];
	MPI_File_seek(file,
		sizeof(unsigned char) + m * sizeof(complexd) * details::rank,
		MPI_SEEK_SET);
	MPI_File_read(file, A, m, MPI_CXX_DOUBLE_COMPLEX,
		MPI_STATUS_IGNORE);
	MPI_File_close(&file);
	return A;
}

static inline int write(const char *f, const complexd *A, uint n)
{
	int status = details::init();
	if (status != 0)
		return status;

	MPI_File file;
	if (MPI_File_open(MPI_COMM_WORLD, f,
		MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file))
	{
		return 4;
	}
	if (!details::rank)
		MPI_File_write(file, (unsigned char*) &n, 1,
			MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
	uint64 m = details::num_of_doubles(n);
	if (m == 0) {
		MPI_File_close(&file);
		return 3;
	}
	MPI_File_seek(file,
		sizeof(unsigned char) + m * sizeof(complexd) * details::rank,
		MPI_SEEK_SET);
	MPI_File_write(file, A, m, MPI_CXX_DOUBLE_COMPLEX,
		MPI_STATUS_IGNORE);
	MPI_File_close(&file);
	return 0;
}


static inline complexd* transformation(const complexd *A, uint n,
	uint i, complexd **P, const uint *k, int *status = nullptr)
{
	int st;
	if (status == nullptr)
		status = &st;

	*status = details::init();
	if (*status != 0)
		return NULL;
	uint64 m = details::num_of_doubles(n);
	if (m == 0) {
		*status = 3;
		return NULL;
	}

	complexd *B = new complexd[m];
	*status = details::trans(A, B, n, i, P, k);
	if (*status != 0) {
		delete [] B;
		return NULL;
	}
	return B;
}

static inline complexd* Hadamard(const complexd *A, uint n, uint k,
	int *status = nullptr)
{
	complexd P0[] = {1/sqrt(2), 1/sqrt(2)};
	complexd P1[] = {1/sqrt(2), -1/sqrt(2)};
	complexd *P[] = {P0, P1};
	return transformation(A, n, 1, P, &k, status);
}

static inline complexd *nHadamard(const complexd *A, uint n,
	int *status = nullptr)
{
	int st;
	if (status == nullptr)
		status = &st;

	*status = details::init();
	if (*status != 0)
		return NULL;
	uint64 m = details::num_of_doubles(n);
	if (m == 0) {
		*status = 3;
		return NULL;
	}

	complexd *B = new complexd[m], *C = new complexd[m], *T,
		*buf = new complexd[m];
	#pragma omp parallel for schedule(guided)
	for (uint64 i = 0; i < m; ++i) {
		B[i] = A[i];
	}

	complexd P0[] = {1/sqrt(2), 1/sqrt(2)};
	complexd P1[] = {1/sqrt(2), -1/sqrt(2)};
	complexd *P[] = {P0, P1};
	for (uint k = 1; k <= n; ++k) {
		*status = details::trans(B, C, n, 1, P, &k, buf);
		if (*status != 0) {
			delete [] B;
			delete [] C;
			delete [] buf;
			return NULL;
		}
		T = B;
		B = C;
		C = T;
	}
	delete [] C;
	delete [] buf;
	return B;
}

static inline complexd* R(const complexd *A, uint n, uint k,
	double a, int *status = nullptr)
{
	complexd P0[] = {1, 0};
	complexd P1[] = {0, cos(a)};
	P1[1].imag(sin(a));
	complexd *P[] = {P0, P1};
	return transformation(A, n, 1, P, &k, status);
}

static inline complexd* NOT(const complexd *A, uint n, uint k,
	int *status = nullptr)
{
	complexd P0[] = {0, 1};
	complexd P1[] = {1, 0};
	complexd *P[] = {P0, P1};
	return transformation(A, n, 1, P, &k, status);
}

static inline complexd* CNOT(const complexd *A, uint n,
	uint k1, uint k2, int *status = nullptr)
{
	uint k[] = {k1, k2};
	complexd P0[] = {1, 0, 0, 0};
	complexd P1[] = {0, 1, 0, 0};
	complexd P2[] = {0, 0, 0, 1};
	complexd P3[] = {0, 0, 1, 0};
	complexd *P[] = {P0, P1, P2, P3};
	return transformation(A, n, 2, P, k, status);
}

static inline complexd* CR(const complexd *A, uint n,
	uint k1, uint k2, double a, int *status = nullptr)
{
	uint k[] = {k1, k2};
	complexd P0[] = {1, 0, 0, 0};
	complexd P1[] = {0, 1, 0, 0};
	complexd P2[] = {0, 0, 1, 0};
	complexd P3[] = {0, 0, 0, cos(a)};
	P3[3].imag(sin(a));
	complexd *P[] = {P0, P1, P2, P3};
	return transformation(A, n, 2, P, k, status);
}

static inline const char* error_comment(int status)
{
	switch (status) {
	case 1: return "Function was called before MPI_init was invoked\n";
	case 2: return "Only for 2^n processes\n";
	case 3: return "Vector too large or too many processes\n";
	case 4: return "Error opening file\n";
	default: return "";
	}
}

}  // namespace QU
#endif  // TASK_4_QUANTUM_H_
