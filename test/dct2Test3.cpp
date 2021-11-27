#include <execution>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

void each_image( std::string input_path, std::string output_path,
	             std::string dictionary_path,
	             std::size_t start_index = 50, std::size_t end_index = 100,
	             std::size_t N1 = 8, std::size_t N2 = 8)
{
	auto input_img = TinyDIP::bmp_read(input_path.c_str(), false);
	//***Load dictionary***
	for (std::size_t i = start_index; i <= end_index; i++)
	{
		std::string fullpath = dictionary_path + "/" + std::to_string(i);
		std::cout << "Dictionary path: " << fullpath << '\n';
		auto input_dbmp = TinyDIP::double_image::read(fullpath.c_str(), false);

	}

}

void dct2Test3( std::string arg1, std::string arg2,
	            std::size_t N1 = 8, std::size_t N2 = 8)
{

}

int main(int argc, char* argv[])
{
	auto arg1 = std::string(argv[1]);
	auto arg2 = std::string(argv[2]);
	dct2Test3(arg1, arg2);
	return 0;
}