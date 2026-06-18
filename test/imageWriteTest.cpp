/* Developed by Jimmy Hu */

#include <algorithm>
#include <cstdlib>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../image_io.h"
#include "../cube.h"
#include "../cube_operations.h"
#include "../timer.h" 

//  imageWriteTest function implementation
void imageWriteTest(const std::size_t width = 32, const std::size_t height = 18)
{
    for (std::size_t y = 0; y < height; ++y)
    {
        TinyDIP::Image<TinyDIP::GrayScale> test_image(width, height);
        for (std::size_t x = 0; x < width; ++x)
        {
            test_image.at(x, y) = 255;
        }
        test_image = TinyDIP::resize_nearest_neighbor(test_image, 1920, 1080);
        test_image = TinyDIP::resize_nearest_neighbor(
            TinyDIP::rotate_detail_shear_transformation_degree(test_image, static_cast<long double>(90)),
            1080,
            1920
        );
        TinyDIP::bmp_write(
            std::string("test_image_") + std::to_string(y),
            TinyDIP::constructRGB(
                test_image,
                test_image,
                test_image
            )
        );
    }
    return;
}

//  generateImage template function implementation
//  Generates a width x height image and sets specific tuple locations to 255
template<std::ranges::random_access_range RangeT>
requires(std::convertible_to<std::ranges::range_value_t<RangeT>, std::tuple<std::size_t, std::size_t>>)
constexpr auto generateImage(
    const RangeT& locations,
    const std::size_t width = 32,
    const std::size_t height = 18
)
{
    TinyDIP::Image<TinyDIP::GrayScale> test_image(width, height);
    
    for (std::size_t y{ 0 }; y < height; ++y)
    {
        for (std::size_t x{ 0 }; x < width; ++x)
        {
            // Use std::ranges::find to check if the current (x, y) is in the provided locations
            if (std::ranges::find(locations, std::tuple<std::size_t, std::size_t>{x, y}) != std::ranges::end(locations))
            {
                test_image.at(x, y) = 255;
            }
            else
            {
                test_image.at(x, y) = 0; // Explicitly set the background to 0
            }
        }
    }
    return test_image;
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    if (argc == 1)
    {
        imageWriteTest();
        return EXIT_SUCCESS;
    }
    return EXIT_SUCCESS;
}