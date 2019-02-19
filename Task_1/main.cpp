#include <iostream>
#include <cstdlib>
#include <complex>
#include <cmath>
#include <omp.h>

using namespace std;

typedef complex<double> complexd;

complexd* gen(int n) {
	
	int m, i;
	m = 1 << n;
	complexd *A = new complexd[m];
	
	double module = 0;
	unsigned int seed = omp_get_wtime();
	
	#pragma omp parallel shared(A, m) firstprivate(seed, i) reduction(+: module)
	{
		seed += omp_get_thread_num();
		#pragma omp for
		for (i = 0; i < m; ++i) {
			A[i].real() = ((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
			A[i].imag() = ((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
			module += abs(A[i]*A[i]);
		}
	}
	for (i = 0; i < m; ++i) A[i] /= sqrt(module);
	
	return A;
	
}

complexd* f(complexd *A, int n, complexd *P, int k) {
	
	int m, l, i;
	m = 1 << n;
	complexd *B = new complexd[m];
	l = 1 << (n - k);
	
	#pragma omp parallel shared(A, B, P, n, k, m, l) private(i)
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
	
	double t = omp_get_wtime();
	complexd *A = gen(n);
	complexd P[] = {1/sqrt(2), 1/sqrt(2), 1/sqrt(2), -1/sqrt(2)};
	complexd *B = f(A, n, P, k);
	
//	for (int i = 0; i < (1 << n); ++i) cout << A[i] << endl;
//	for (int i = 0; i < (1 << n); ++i) cout << B[i] << endl;
	
	cout << '~' << n << '\t' << k << '\t' << getenv("OMP_NUM_THREADS") << '\t' << omp_get_wtime() - t << endl;
	free(A);
	free(B);
	return 0;
	
}
