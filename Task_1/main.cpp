#include <iostream>
#include <cstdlib>
#include <complex>
#include <cmath>
#include <omp.h>

using namespace std;

typedef complex<double> complexd;

complexd* gen(int n) {
	
	unsigned long long i, m = 1LLU << n;
	complexd *A = new complexd[m];
	
	double module = 0;
	unsigned int seed = omp_get_wtime();
	#pragma omp parallel shared(A, m) firstprivate(seed) private(i) reduction(+: module)
	{
		seed += omp_get_thread_num();
		#pragma omp for
		for (i = 0; i < m; ++i) {
			A[i].real() = ((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
			A[i].imag() = ((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
			module += abs(A[i]*A[i]);
		}
	}
	module = sqrt(module);
	
	#pragma omp parallel for
	for (i = 0; i < m; ++i) A[i] /= module;
	
	return A;
	
}

complexd* f(complexd *A, int n, complexd *P, int k) {
	
	unsigned long long i, m = 1LLU << n, l = 1LLU << (n - k);
	complexd *B = new complexd[m];
	
	#pragma omp parallel shared(A, B, P, m, l) private(i)
	{
		#pragma omp for
		for (i = 0; i < m; ++i)
			B[i] = ((i & l) == 0) ? P[0]*A[i & ~l] + P[1]*A[i | l] : P[2]*A[i & ~l] + P[3]*A[i | l];
	}
	
	return B;
	
}

int main(int argc, char **argv) {
	
	if (argc < 3) {
		cout << argv[0] << " n k" << endl;
		return 0;
	}
	
	int n, k;
	n = atoi(argv[1]);
	k = atoi(argv[2]);
	
	double t0 = omp_get_wtime();
	complexd *A = gen(n);
	double t1 = omp_get_wtime();
	complexd P[] = {1/sqrt(2), 1/sqrt(2), 1/sqrt(2), -1/sqrt(2)};
	complexd *B = f(A, n, P, k);
	double t2 = omp_get_wtime();
	
	/*
	unsigned long long i, m = 1LLU << n;
	for (i = 0; i < m; ++i) cout << A[i] << endl;
	cout << endl;
	for (i = 0; i < m; ++i) cout << B[i] << endl;
	*/
	
	cout << '~' << n << '\t' << k << '\t' << getenv("OMP_NUM_THREADS") << '\t' << t1 - t0 << '\t' << t2 - t1 << endl;
	free(A);
	free(B);
	return 0;
	
}
