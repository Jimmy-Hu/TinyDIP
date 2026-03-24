//  Developed by Jimmy Hu

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

//  localDimmingTest Function Implementation
static auto localDimmingTest(const std::filesystem::path& input_path, const std::string_view output_path)
{
    auto input_img = TinyDIP::bmp_read(input_path.string().c_str(), true);
    auto RGB_max_result = RGB_max(input_img);
    TinyDIP::bmp_write(std::string(output_path).c_str(), RGB_max_result);
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    std::cout << "argc = " << std::to_string(argc) << '\n';
    if(argc == 2)
    {
        auto input_path = std::string(argv[1]);
        auto input_img = TinyDIP::bmp_read(input_path.c_str(), true);
        auto RGB_max_result = RGB_max(input_img);
        TinyDIP::bmp_write("RGB_max_result", RGB_max_result);
    }
    else if  (argc == 3)
    {
        auto input_path = std::string(argv[1]);
        auto output_path = std::string(argv[2]);
        auto input_img = TinyDIP::bmp_read(input_path.c_str(), true);
        auto RGB_max_result = RGB_max(input_img);
        TinyDIP::bmp_write(output_path.c_str(), RGB_max_result);
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " <input_image_path> [output_image_path]\n";
    }
    return EXIT_SUCCESS;
}



