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