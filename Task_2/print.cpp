#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char **argv) {
	
	if (argc < 2) {
		cout << argv[0] << " file" << endl;
		return 0;
	}
	
	ifstream file(argv[1], ios::binary);
	if (!file.is_open()) {
		cout << "Error opening file " << argv[1] << endl;
		return 1;
	}
	
	int n;
	file.read((char*)&n, sizeof(int));
	cout << "2^" << n << endl << scientific;
	
	unsigned long long m = 1LLU << n, i;
	double d[2];
	for (i = 0; i < m; ++i) {
		file.read((char*)d, 2*sizeof(double));
		cout << d[0] << '\t' << d[1] << endl;
	}
	
	file.close();
	return 0;
	
}
