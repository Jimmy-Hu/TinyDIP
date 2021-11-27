#include <execution>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

void each_image( std::size_t start_index = 50, std::size_t end_index = 100,
	             std::size_t N1 = 8, std::size_t N2 = 8)
{

}

void dct2Test3( std::string arg1, std::string arg2)
{

}

int main(int argc, char* argv[])
{
	auto arg1 = std::string(argv[1]);
	auto arg2 = std::string(argv[2]);
	dct2Test3(arg1, arg2);
	return 0;
}