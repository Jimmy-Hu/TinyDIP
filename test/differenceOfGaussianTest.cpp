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

void differenceOfGaussianTest(std::string_view input_image_path = "InputImages/1", std::string_view output_image_path = "OutputImages/differenceOfGaussianTest")
{
    auto input_img = TinyDIP::bmp_read(std::string(input_image_path).c_str(), false);
    for(int sigma = 1; sigma < 10; ++sigma)
    {
        auto output_img = TinyDIP::im2uint8(
                                TinyDIP::multiplies(
                                    TinyDIP::abs(
                                        TinyDIP::difference_of_gaussian(TinyDIP::im2double(input_img), static_cast<double>(sigma), 1.0)
                                    ),
                                    3
                                )
                            );
        TinyDIP::bmp_write(
            (std::string(output_image_path) + std::string("_sigma=") + std::to_string(sigma)).c_str(),
            output_img);
    }
    return;
}

int main(int argc, char* argv[])
{
    auto start = std::chrono::system_clock::now();
    differenceOfGaussianTest();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}