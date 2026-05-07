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

void imageWriteTest(const std::size_t width = 32, const std::size_t height = 18)
{
    TinyDIP::Image<TinyDIP::GrayScale> test_image(width, height);
    test_image.at(10, 10) = 255;
    test_image = TinyDIP::lanczos_resample(std::execution::par, test_image, 1920, 1080);
    TinyDIP::bmp_write(
        "test_image",
        TinyDIP::constructRGB(
            test_image,
            test_image,
            test_image
        )
    );
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