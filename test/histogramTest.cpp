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
    auto histogram_result1 = TinyDIP::histogram(TinyDIP::getVplane(hsv_image));
    os << "*****  std::map Histogram  *****\n";
    for (const auto& [key, value] : histogram_result1 )
    {
        os << "key = " << key << ", value = " << value << '\n';
    }
    os << "*****  Normalized std::map Histogram  *****\n";
    auto normalized_histogram1 = TinyDIP::normalize_histogram(histogram_result1);
    double sum = 0.0;
    for (const auto& [key, value] : normalized_histogram1)
    {
        os << "key = " << key << ", value = " << value << '\n';
        sum += value;
    }
    os << "sum = " << sum << '\n';
    os << "-------------------------------------------------------";
    auto histogram_result2 = TinyDIP::histogram(TinyDIP::im2uint8(TinyDIP::getVplane(hsv_image)));
    os << "*****  std::array Histogram  *****\n";
    for (std::size_t i = 0; i < histogram_result2.size(); ++i)
    {
        std::cout << i << " count = " << histogram_result2[i] << '\n';
    }
    auto normalized_histogram2 = TinyDIP::normalize_histogram(execution_policy, histogram_result2);
    os << "*****  Normalized std::array Histogram  *****\n";
    sum = 0.0;
    for (std::size_t i = 0; i < normalized_histogram2.size(); ++i)
    {
        std::cout << i << " count = " << normalized_histogram2[i] << '\n';
        sum += normalized_histogram2[i];
    }
    os << "sum = " << sum << '\n';
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
    HistogramTest(std::execution::par, image_input);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}
