#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include "quantum.h"

int main(int argc, char **argv)
{
	if (argc < 3) {
		std::cout << argv[0] << " A B [epsilon]" << std::endl;
		return 0;
	}

	float epsilon = 1.e-4;
	if (argc >= 4)
		sscanf(argv[3], "%f", &epsilon);

	std::ifstream f1(argv[1], std::ios::binary),
		f2(argv[2], std::ios::binary);
	if (!f1.is_open()) {
		std::cout << "Error opening file " << argv[1] << std::endl;
		return 1;
	}
	if (!f2.is_open()) {
		std::cout << "Error opening file " << argv[2] << std::endl;
		return 1;
	}

	unsigned char n, n1;
	f1.read(reinterpret_cast<char*>(&n), sizeof(unsigned char));
	f2.read(reinterpret_cast<char*>(&n1), sizeof(unsigned char));
	if (n != n1) {
		std::cout << '0' << std::endl;
		std::cout << "Vectors of different dimensions" << std::endl;
		return 0;
	}

	unsigned long long m = 1LLU << n, i;
	QU::complexd x, y;
	for (i = 0; i < m; ++i) {
		f1.read(reinterpret_cast<char*>(&x), sizeof(QU::complexd));
		f2.read(reinterpret_cast<char*>(&y), sizeof(QU::complexd));
		if (abs(x - y) > epsilon) {
			std::cout << '0' << std::endl;
			std::cout << i << "): " << std::scientific <<
				x.real() << " + i* " << x.imag() << " != " <<
				y.real() << " + i* " << y.imag() << std::endl;
			f1.close();
			f2.close();
			return 0;
		}
	}

	std::cout << '1' << std::endl;
	f1.close();
	f2.close();
	return 0;
}
