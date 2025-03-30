/* Developed by Jimmy Hu */
#include <chrono>
#include <ostream>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"

template<class ExPo, class ElementT>
requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
constexpr static auto HistogramTest(
    ExPo execution_policy,
    const TinyDIP::Image<ElementT>& input,
    std::ostream& os = std::cout
)
{
    auto hsv_image = TinyDIP::rgb2hsv(execution_policy, input);
    auto v_plane = TinyDIP::getVplane(hsv_image);
    TinyDIP::Timer timer1;
    os << "*****  histogram of the image  *****\n";
    auto histogram_result = TinyDIP::histogram(v_plane);
    double sum = 0.0;
    for (const auto& [key, value] : histogram_result)
    {
        os << "key = " << key << ", value = " << value << '\n';
        sum += value;
    }
    os << "sum = " << sum << '\n';
    os << "-------------------------------------------------------\n";
    auto histogram_with_bins_output =
        TinyDIP::histogram_with_bins(v_plane, 0.0, 255.0);
    os << "*****  histogram_with_bins Results  *****\n";
    for (std::size_t i = 0; i < histogram_with_bins_output.size(); i++)
    {
        os << "Bin index = " << i << " Value = " << histogram_with_bins_output[i] << "\n";
    }
    
    return;
}

int main()
{
    TinyDIP::Timer timer1;
    std::string image_filename = "../InputImages/1.bmp";
    auto image_input = TinyDIP::bmp_read(image_filename.c_str(), true);
    image_input = TinyDIP::copyResizeBicubic(image_input, 3 * image_input.getWidth(), 3 * image_input.getHeight());
    HistogramTest(std::execution::par, image_input);
    return EXIT_SUCCESS;
}
