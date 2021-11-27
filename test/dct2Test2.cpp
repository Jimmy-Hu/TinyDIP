#include <execution>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

void dct2Test2( std::string arg1, std::string arg2,
                std::string arg3,
                std::size_t N1 = 8, std::size_t N2 = 8)
{
	std::cout << "dct2Test2 program..." << '\n';
	std::cout << arg1 << '\n';
	std::cout << arg2 << '\n';
	std::size_t start_index = 50, end_index = 100;
	for (std::size_t i = start_index; i <= end_index; i++)
	{
		std::string fullpath = arg1 + "/" + std::to_string(i);
		std::cout << fullpath << '\n';
		auto input_img = TinyDIP::bmp_read(fullpath.c_str(), false);
		auto dct2_results = TinyDIP::recursive_transform<2>(
			std::execution::par,
			[](auto&& element) { return TinyDIP::dct2(element); },
			TinyDIP::split(TinyDIP::getVplane(TinyDIP::rgb2hsv(input_img)), input_img.getWidth() / N1, input_img.getHeight() / N2)
			);
		auto dct2_combined = TinyDIP::concat(dct2_results);
		TinyDIP::double_image::write(arg3.c_str(), dct2_combined);
	}
	return;
}

int main(int argc, char* argv[])
{
	auto arg1 = std::string(argv[1]);
	auto arg2 = std::string(argv[2]);
	auto arg3 = std::string(argv[3]);
	dct2Test2(arg1, arg2, arg3);
	return 0;
}

