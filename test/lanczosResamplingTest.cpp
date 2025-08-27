/* Developed by Jimmy Hu */

#include <chrono>
#include <execution>
#include <map>
#include <omp.h>
#include <sstream>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"

void lanczosResamplingTest(
    std::string_view input_image_path = "../InputImages/1",
    std::string_view output_image_path = "../OutputImages/lanczosResamplingTest")
{
    auto input_img = TinyDIP::bmp_read(std::string(input_image_path).c_str(), false);
    auto output_img =
        TinyDIP::lanczos_resample(
            input_img,
            input_img.getWidth() * 2,
            input_img.getHeight() * 2
        );
    TinyDIP::bmp_write(
        std::string(output_image_path).c_str(),
        output_img);
    
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer;
    lanczosResamplingTest();
    return EXIT_SUCCESS;
}