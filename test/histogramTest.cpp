/* Developed by Jimmy Hu */
#include <chrono>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

template<class ExPo, class ElementT>
requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
constexpr static auto HistogramTest(
	ExPo execution_policy,
	const TinyDIP::Image<ElementT>& input,
	std::ostream& os = std::cout
)
{
	auto hsv_image = TinyDIP::rgb2hsv(execution_policy, input);
	auto start1 = std::chrono::system_clock::now();
	auto histogram_result1 = TinyDIP::histogram_normalized(TinyDIP::im2uint8(TinyDIP::getVplane(hsv_image)));
	auto end1 = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds1 = end1 - start1;
	os << "elapsed time: " << elapsed_seconds1.count() << '\n';
	return histogram_result1;
}

int main()
{
	auto start = std::chrono::system_clock::now();
	std::string image_filename = "../InputImages/DiamondWheelTool/1.bmp";
	auto image_input = TinyDIP::bmp_read(image_filename.c_str(), true);
	image_input = TinyDIP::copyResizeBicubic(image_input, 3 * image_input.getWidth(), 3 * image_input.getHeight());
	auto histogram_result1 = HistogramTest(std::execution::par, image_input);
	double sum = 0.0;
	for (std::size_t i = 0; i < histogram_result1.size(); ++i)
	{
		std::cout << i << " count: " << histogram_result1[i] << "\n";
		sum += histogram_result1[i];
	}
	std::cout << "Sum = " << sum << '\n';
	std::cout << "image_input.count = " << image_input.count() << '\n';
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return EXIT_SUCCESS;
}
