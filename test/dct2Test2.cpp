#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void dct2Test2(std::string arg1)
{
	std::cout << "dct2Test2 program..." << '\n';
	std::cout << arg1 << '\n';
}

int main(int argc, char* argv[])
{
	auto arg1 = std::string(argv[0]);
	dct2Test2(arg1);
	return 0;
}

