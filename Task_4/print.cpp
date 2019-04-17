#include <iostream>
#include <fstream>
#include "quantum.h"

int main(int argc, char **argv)
{
	if (argc < 2) {
		std::cout << argv[0] << " file" << std::endl;
		return 0;
	}

	std::ifstream file(argv[1], std::ios::binary);
	if (!file.is_open()) {
		std::cout << "Error opening file " << argv[1] << std::endl;
		return 1;
	}
	unsigned char n;
	file.read(reinterpret_cast<char*>(&n), sizeof(unsigned char));
	std::cout << "2^" << static_cast<int>(n) << std::endl <<
		std::scientific;

	unsigned long long m = 1LLU << n, i;
	QU::complexd c;
	for (i = 0; i < m; ++i) {
		file.read(reinterpret_cast<char*>(&c), sizeof(QU::complexd));
		std::cout << c.real() << '\t' << c.imag() << std::endl;
	}

	file.close();
	return 0;
}
