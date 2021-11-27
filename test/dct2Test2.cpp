#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

void dct2Test2(std::string arg1, std::string arg2)
{
	std::cout << "dct2Test2 program..." << '\n';
	std::cout << arg1 << '\n';
	std::cout << arg2 << '\n';
	std::size_t start_index = 50, end_index = 100;
	for (std::size_t i = start_index; i < end_index; i++)
	{
		std::string fullpath = arg1 + "/" + std::to_string(i);
		std::cout << fullpath << '\n';
		TinyDIP::bmp_read(fullpath.c_str(), false);
	}
	return;
}

int main(int argc, char* argv[])
{
	auto arg1 = std::string(argv[1]);
	auto arg2 = std::string(argv[2]);
	dct2Test2(arg1, arg2);
	return 0;
}

