#include <execution>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

template<class ElementT>
constexpr static auto get_offset( TinyDIP::Image<ElementT>& input,
	                              std::vector<TinyDIP::Image<double>>& dictionary_x,
	                              std::vector<TinyDIP::Image<double>>& dictionary_y)
{
	return input;
}

void each_image( std::string input_path, std::string output_path,
	             std::vector<TinyDIP::Image<double>>& dictionary_x,
	             std::vector<TinyDIP::Image<double>>& dictionary_y,
	             std::size_t N1 = 8, std::size_t N2 = 8)
{
	auto input_img = TinyDIP::bmp_read(input_path.c_str(), false);
	auto input_hsv = TinyDIP::rgb2hsv(input_img);
	auto h_plane = TinyDIP::getHplane(input_hsv);
	auto s_plane = TinyDIP::getSplane(input_hsv);
	auto v_plane = TinyDIP::getVplane(input_hsv);
	auto input_dct_blocks = TinyDIP::recursive_transform<2>(
		std::execution::par,
		[](auto&& element) { return TinyDIP::dct2(element); },
		TinyDIP::split(v_plane, v_plane.getWidth() / N1, v_plane.getHeight() / N2)
		);

	auto output_dct_blocks = TinyDIP::recursive_transform<2>(
		std::execution::par,
		[&](auto&& element) { return get_offset(element, dictionary_x, dictionary_y); },
		input_dct_blocks
		);
	auto output_img = TinyDIP::hsv2rgb(TinyDIP::constructHSV(
		h_plane,
		s_plane,
		TinyDIP::concat(
			TinyDIP::recursive_transform<2>(
				std::execution::par,
				[](auto&& element) { return TinyDIP::dct2(element); },
				output_dct_blocks))
	));
	TinyDIP::bmp_write(output_path.c_str(), output_img);
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
		std::string output_path = output_folder + "/" + std::to_string(i);
		each_image(fullpath, output_path, x, y);
	}
	return;
}

int main(int argc, char* argv[])
{
	auto arg1 = std::string(argv[1]);
	auto arg2 = std::string(argv[2]);
	auto arg3 = std::string(argv[3]);
	dct2Test3(arg1, arg2, arg3);
	return 0;
}