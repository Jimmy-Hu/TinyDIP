/* Developed by Jimmy Hu */
#include <chrono>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"
#include <tbb/parallel_for.h>

template<class ExPo, class ElementT>
requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
constexpr static auto DiamondWheelAnalysisTest(
	ExPo execution_policy,
	const TinyDIP::Image<ElementT>& input,
	std::ostream& os = std::cout
)
{
	auto hsv_image = TinyDIP::rgb2hsv(execution_policy, input);
	auto start1 = std::chrono::system_clock::now();
	auto processed_hsv_image1 = TinyDIP::apply_each_pixel(
		std::execution::par,
		hsv_image,
		[&](TinyDIP::HSV pixel) -> TinyDIP::HSV
		{
			TinyDIP::HSV pixel_for_filling;
			pixel_for_filling.channels[0] = 0;
			pixel_for_filling.channels[1] = 1;
			pixel_for_filling.channels[2] = 255;
			if (pixel.channels[2] > 100 && pixel.channels[2] < 220)
			{
				pixel = pixel_for_filling;
			}
			return pixel;
		}
		);
	auto end1 = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds1 = end1 - start1;
	os << "elapsed time with using parallel execution policy: " << elapsed_seconds1.count() << '\n';
	auto start2 = std::chrono::system_clock::now();
	auto processed_hsv_image2 = TinyDIP::apply_each_pixel_openmp(hsv_image, 
		[&](TinyDIP::HSV pixel) -> TinyDIP::HSV
		{
			TinyDIP::HSV pixel_for_filling;
			pixel_for_filling.channels[0] = 0;
			pixel_for_filling.channels[1] = 1;
			pixel_for_filling.channels[2] = 255;
			if (pixel.channels[2] > 100 && pixel.channels[2] < 220)
			{
				pixel = pixel_for_filling;
			}
			return pixel;
		});
	auto end2 = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
	os << "elapsed time with using Openmp: " << elapsed_seconds2.count() << '\n';
	os << TinyDIP::sum(TinyDIP::difference(processed_hsv_image1, processed_hsv_image2)) << '\n';
	auto output_image = TinyDIP::hsv2rgb(execution_policy, processed_hsv_image1);
	return output_image;
}

int main()
{
	TinyDIP::Timer timer1;
	std::string image_filename = "../InputImages/DiamondWheelTool/1.bmp";
	auto image_input = TinyDIP::bmp_read(image_filename.c_str(), true);
	image_input = TinyDIP::copyResizeBicubic(image_input, 3 * image_input.getWidth(), 3 * image_input.getHeight());
	TinyDIP::bmp_write("BeforeProcessing", image_input);
	auto output_image = DiamondWheelAnalysisTest(std::execution::seq, image_input);
	TinyDIP::bmp_write("AfterProcessing", output_image);
	return EXIT_SUCCESS;
}
