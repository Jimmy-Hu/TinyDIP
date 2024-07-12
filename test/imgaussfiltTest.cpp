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

void imgaussfiltTest(std::string_view input_image_path = "InputImages/1", std::string_view output_image_path = "OutputImages/imgaussfiltTest")
{
    auto input_img = TinyDIP::bmp_read(std::string(input_image_path).c_str(), false);
    auto output_img = TinyDIP::im2uint8(TinyDIP::imgaussfilt(TinyDIP::im2double(input_img)));
    TinyDIP::bmp_write(std::string(output_image_path).c_str(), output_img);
}

int main(int argc, char* argv[])
{
    auto start = std::chrono::system_clock::now();
    imgaussfiltTest();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}