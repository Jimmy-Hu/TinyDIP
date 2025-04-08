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

//  remove_extension Function Implementation
//  Copy from: https://stackoverflow.com/a/6417908/6667035
std::string remove_extension(const std::string& filename)
{
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    omp_set_num_threads(18); // Use 18 threads for all consecutive parallel regions
    std::cout << "argc parameter: " << std::to_string(argc) << '\n';
    if(argc == 3)
    {
        auto input_path = std::string(argv[1]);
        auto input_img = TinyDIP::bmp_read(input_path.c_str(), true);
        
        auto output_path = std::string(argv[2]);
        std::cout << "Save output to " << output_path << '\n';
        TinyDIP::bmp_write(output_path.c_str(), input_img);
    }
    else
    {
        std::string input_path = "../InputImages/1.bmp";
        auto input_img = TinyDIP::bmp_read(input_path.c_str(), true);
        input_img = TinyDIP::copyResizeBicubic(input_img, input_img.getWidth() * 3, input_img.getHeight() * 3);
        auto output_img = TinyDIP::bilateral_filter(
            std::execution::seq,
            input_img,
            20,
            [](double input) { return TinyDIP::normalDistribution1D(input, 3.0); },
            [](double input) { return TinyDIP::normalDistribution1D(input, 3.0); });
        std::string output_path = "output_test";
        std::cout << "Save output to " << output_path << '\n';
        TinyDIP::bmp_write(output_path.c_str(), output_img);
        std::string difference_output_path = "difference_output";
        std::cout << "Save output to " << difference_output_path << '\n';
        TinyDIP::bmp_write(
            difference_output_path.c_str(),
            TinyDIP::increase_intensity(TinyDIP::difference(input_img, output_img), static_cast<TinyDIP::GrayScale>(50))
        );
    }
    return EXIT_SUCCESS;
}