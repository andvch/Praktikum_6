#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

using namespace std;

int main(int argc, char **argv) {
	
	if (argc < 3) {
		cout << argv[0] << " A B [epsilon]" << endl;
		return 0;
	}
	
	float epsilon = 1.e-4;
	if (argc >= 4) epsilon = atof(argv[3]);
	
	ifstream f1(argv[1], ios::binary), f2(argv[2], ios::binary);
	if (!f1.is_open()) {
		cout << "Error opening file " << argv[1] << endl;
		return 1;
	}
	if (!f2.is_open()) {
		cout << "Error opening file " << argv[2] << endl;
		return 1;
	}
	
	int n, n1;
	f1.read((char*)&n, sizeof(int));
	f2.read((char*)&n1, sizeof(int));
	if (n != n1) {
		cout << '0' << endl;
		cout << "Вектора разных размерностей" << endl;
		return 0;
	}
	
	unsigned long long m = 1LLU << n, i;
	double x[2], y[2];
	for (i = 0; i < m; ++i) {
		f1.read((char*)&x, 2*sizeof(double));
		f2.read((char*)&y, 2*sizeof(double));
		if ((abs(x[0]-y[0]) > epsilon) or (abs(x[1]-y[1]) > epsilon)) {
			cout << '0' << endl;
			cout << i << "): " << scientific << x[0] << " + i* " << x[1] << " != " << y[0] << " + i* " << y[1] << endl;
			f1.close(); f2.close(); return 0;
		}
	}
	
	cout << '1' << endl;
	f1.close();
	f2.close();
	return 0;
	
}
