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

void imgaussfiltTest(std::string_view input_image_path = "InputImages/1", std::string_view output_image_path = "OutputImages/imgaussfiltTest")
{
    auto input_img = TinyDIP::bmp_read(std::string(input_image_path).c_str(), false);
    for(int sigma = 1; sigma < 10; ++sigma)
    {
        auto output_img_mirror = TinyDIP::im2uint8(
                                TinyDIP::imgaussfilt(
                                    TinyDIP::im2double(input_img),
                                    static_cast<double>(sigma),
                                    static_cast<int>(TinyDIP::computeFilterSizeFromSigma(sigma)),
                                    TinyDIP::BoundaryCondition::mirror)
                                );
        auto output_img_replicate = TinyDIP::im2uint8(
                                TinyDIP::imgaussfilt(
                                    TinyDIP::im2double(input_img),
                                    static_cast<double>(sigma),
                                    static_cast<int>(TinyDIP::computeFilterSizeFromSigma(sigma)),
                                    TinyDIP::BoundaryCondition::replicate)
                                );
        TinyDIP::bmp_write(
            (std::string(output_image_path) + std::string("_sigma=") + std::to_string(sigma) + std::string("_mirror")).c_str(),
            output_img_mirror);
        TinyDIP::bmp_write(
            (std::string(output_image_path) + std::string("_sigma=") + std::to_string(sigma) + std::string("_replicate")).c_str(),
            output_img_replicate);
    }
    
}

void imgaussfiltTest2(
    std::string_view input_image_path = "../InputImages/1",
    std::string_view output_image_path = "../OutputImages/imgaussfiltTest2")
{
    auto input_img = TinyDIP::bmp_read(std::string(input_image_path).c_str(), false);
    auto output_img = 
        TinyDIP::imgaussfilt(
            std::execution::par,
            input_img,
            20.0,
            20.0,
            1.0,
            50,
            50,
            TinyDIP::constant
            );
    /*
    auto output_img = TinyDIP::im2uint8(
        TinyDIP::imgaussfilt(
            std::execution::par,
            TinyDIP::im2double(input_img),
            512,
            512,
            500.0,
            500.0,
            0.7,
            1.0,
            TinyDIP::constant)
    );
    */
    TinyDIP::bmp_write(
        (std::string(output_image_path)).c_str(),
        output_img);
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer;
    imgaussfiltTest2();
    return EXIT_SUCCESS;
}