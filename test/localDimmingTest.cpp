//  Developed by Jimmy Hu

#include <algorithm>
#include <cassert>
#include <execution>
#include <filesystem>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../image_io.h"
#include "../cube.h"
#include "../cube_operations.h"
#include "../timer.h"

//  RGB_max Function Implementation
static auto RGB_max(const TinyDIP::Image<TinyDIP::RGB>& input_image)
{
    return TinyDIP::pixelwise_transform([](auto&& each_pixel)
            {
                auto max_value = std::ranges::max(each_pixel.channels);
                TinyDIP::RGB new_pixel{ max_value, max_value, max_value };
                return new_pixel;
            }, input_image);
}

//  RGB_max_parallel Function Implementation
static auto RGB_max_parallel(const TinyDIP::Image<TinyDIP::RGB>& input_image)
{
    return TinyDIP::pixelwise_transform(std::execution::par_unseq, [](auto&& each_pixel)
            {
                auto max_value = std::ranges::max(each_pixel.channels);
                TinyDIP::RGB new_pixel{ max_value, max_value, max_value };
                return new_pixel;
            }, input_image);
}

//  belongs_bin_index Function Implementation
template<std::ranges::random_access_range RangeT, class ElementT>
requires(std::equality_comparable<std::ranges::range_value_t<RangeT>> and
         std::equality_comparable<ElementT> and
         std::convertible_to<std::ranges::range_value_t<RangeT>, ElementT>)
static auto belongs_bin_index(const RangeT& thresholds, const ElementT& value)
{
    auto it = std::ranges::lower_bound(thresholds, value);
    return static_cast<int>(std::distance(std::ranges::begin(thresholds), it)) - 1;
}

//  gray2gamma Template Function Implementation
//  Output is TinyDIP::Image<TinyDIP::RGB_DOUBLE>, 12 bits
template<
    class ExecutionPolicy,
    std::ranges::random_access_range GammaRange1,
    std::ranges::random_access_range GammaRange2>
requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> and
         std::equality_comparable<std::ranges::range_value_t<GammaRange1>> and
         std::equality_comparable<std::ranges::range_value_t<GammaRange2>>)
static auto gray2gamma(
    ExecutionPolicy&& policy,
    const TinyDIP::Image<TinyDIP::RGB>& input_image,
    const GammaRange1& gamma_range1,
    const GammaRange2& gamma_range2
)
{
    return TinyDIP::pixelwise_transform(std::forward<ExecutionPolicy>(policy), [&](auto&& each_pixel)
            {
                auto pixel_value = each_pixel.channels[0];
                auto bin_index = belongs_bin_index(gamma_range1, pixel_value);
                auto final_pixel_value = std::clamp(
                    gamma_range2[bin_index] + (((gamma_range2[bin_index + 1] - gamma_range2[bin_index]) * (pixel_value - gamma_range1[bin_index])) >> (static_cast<int>(std::log2(gamma_range1[bin_index + 1] - gamma_range1[bin_index])))),
                    0, static_cast<int>(std::pow(2, 12) - 1)
                );
                //final_pixel_value = final_pixel_value >> 4;         //  12 bits to 8 bits
                TinyDIP::RGB_DOUBLE new_pixel{ final_pixel_value, final_pixel_value, final_pixel_value };
                return new_pixel;
            }, input_image);
}

//  localDimmingTest Function Implementation
static auto localDimmingTest(
    const std::filesystem::path& input_path,
    const std::string_view output_path,
    const std::size_t light_bead_width = 22,
    const std::size_t light_bead_height = 8,
    const std::size_t x_extension_pixel_count = 41,
    const std::size_t y_extension_pixel_count = 45
)
{
    auto input_img = TinyDIP::bmp_read(input_path.string().c_str(), true);
    auto RGB_max_result = RGB_max(input_img);
    std::vector<int> gamma_node = {0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256};
    std::vector<int> gamma_vale = {0, 2, 9, 23, 43, 69, 104, 146, 195, 253, 320, 394, 477, 569, 670, 780, 899, 1026, 1165, 1312, 1468, 1635, 1810, 1997, 2193, 2399, 2615, 2842, 3079, 3326, 3584, 3851, 4130};
    auto output_image_12bits = gray2gamma(std::execution::par_unseq, RGB_max_result, gamma_node, gamma_vale);
    auto split_overlap_output = TinyDIP::split_overlap(
        std::execution::par_unseq,
        light_bead_width,
        light_bead_height,
        x_extension_pixel_count,
        y_extension_pixel_count
    );
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    std::cout << "argc = " << std::to_string(argc) << '\n';
    if(argc == 2)
    {
        std::filesystem::path input_path = std::string(argv[1]);
        if (!std::filesystem::exists(input_path))
        {
            std::cerr << "File not found: " << input_path << '\n';
            return EXIT_SUCCESS;
        }
        localDimmingTest(input_path, std::string("localDimmingTest"));
    }
    else if  (argc == 3)
    {
        std::filesystem::path input_path = std::string(argv[1]);
        std::filesystem::path output_path = std::string(argv[2]);
        if (!std::filesystem::exists(input_path))
        {
            std::cerr << "File not found: " << input_path << '\n';
            return EXIT_SUCCESS;
        }
        std::filesystem::path path_without_extension = output_path.parent_path() / output_path.stem();
        localDimmingTest(input_path, path_without_extension.string());
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " <input_image_path> [output_image_path]\n";
    }
    return EXIT_SUCCESS;
}



