/* Developed by Jimmy Hu */
#include <chrono>
#include <filesystem>
#include <ostream>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"

template<class ExPo, class ElementT>
requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
constexpr static auto otsuThresholdTest(
    ExPo execution_policy,
    const TinyDIP::Image<ElementT>& input,
    std::ostream& os = std::cout
)
{
    auto hsv_image = TinyDIP::rgb2hsv(execution_policy, input);
    TinyDIP::Timer timer1;
    auto unit8_image = TinyDIP::im2uint8(TinyDIP::getVplane(hsv_image));
    return TinyDIP::apply_threshold_openmp(unit8_image, TinyDIP::otsu_threshold(execution_policy, unit8_image));
}

int main(int argc, char* argv[])
{
    std::string image_filename = "../InputImages/1.bmp";    //  Default file path
    if (argc == 2)                                          //  User has specified input file
    {
        image_filename = std::string(argv[1]);
    }
    if (!std::filesystem::is_regular_file(image_filename))
    {
        throw std::runtime_error("Error: File not found!");
    }
    TinyDIP::Timer timer1;
    auto image_input = TinyDIP::bmp_read(image_filename.c_str(), true);
    image_input = TinyDIP::copyResizeBicubic(image_input, 3 * image_input.getWidth(), 3 * image_input.getHeight());
    auto image_output = otsuThresholdTest(std::execution::par, image_input);
    TinyDIP::bmp_write("test_output", TinyDIP::constructRGB(image_output, image_output, image_output));
    return EXIT_SUCCESS;
}
