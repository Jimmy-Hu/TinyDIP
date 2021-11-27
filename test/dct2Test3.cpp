#include <execution>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

void each_image( std::string input_path, std::string output_path,
	             std::vector<TinyDIP::Image<double>>& dictionary_x,
	             std::vector<TinyDIP::Image<double>>& dictionary_y,
	             std::size_t N1 = 8, std::size_t N2 = 8)
{
	auto input_img = TinyDIP::bmp_read(input_path.c_str(), false);
	

}

void dct2Test3( std::string input_folder, std::string output_folder,
	            std::string dictionary_path,
	            std::size_t start_index = 1, std::size_t end_index = 1,
	            std::size_t dic_start_index = 50, std::size_t dic_end_index = 100,
	            std::size_t N1 = 8, std::size_t N2 = 8)
{
	std::cout << "dct2Test3 program..." << '\n';
	//***Load dictionary***
	std::vector<TinyDIP::Image<double>> x, y;
	for (std::size_t i = dic_start_index; i <= dic_end_index; i++)
	{
		std::string fullpath = dictionary_path + "/" + std::to_string(i);
		std::cout << "Dictionary path: " << fullpath << '\n';
		auto input_dbmp = TinyDIP::double_image::read(fullpath.c_str(), false);
		auto dct_block_x = TinyDIP::split(input_dbmp, input_dbmp.getWidth() / N1, input_dbmp.getHeight() / N2);
		TinyDIP::recursive_for_each<2>(dct_block_x, [&](auto&& element) { x.push_back(element); });

		std::string fullpath_gt = dictionary_path + "/GT";
		auto input_dbmp_gt = TinyDIP::double_image::read(fullpath_gt.c_str(), false);
		auto dct_block_y = TinyDIP::split(input_dbmp_gt, input_dbmp_gt.getWidth() / N1, input_dbmp_gt.getHeight() / N2);
		TinyDIP::recursive_for_each<2>(dct_block_y, [&](auto&& element) { y.push_back(element); });
	}
	std::cout << "x count: " << x.size() << "\ty count: " << y.size() << '\n';
	
	for (std::size_t i = start_index; i < end_index; i++)
	{
		std::string fullpath = input_folder + "/" + std::to_string(i);
		std::cout << "fullpath: " << fullpath << '\n';
		each_image(fullpath, output_folder, x, y);

	}
}

int main(int argc, char* argv[])
{
	auto arg1 = std::string(argv[1]);
	auto arg2 = std::string(argv[2]);
	auto arg3 = std::string(argv[3]);
	dct2Test3(arg1, arg2, arg3);
	return 0;
}