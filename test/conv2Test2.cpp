/* Developed by Jimmy Hu */

#include <cassert>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

int main()
{
    auto start = std::chrono::system_clock::now();
    
    auto image1 = TinyDIP::bmp_read("InputImages/1", false);
    std::size_t mask_size = 5;
    std::vector<double> mask_data;
    for (std::size_t i = 0; i < mask_size * mask_size; ++i)
    {
        mask_data.emplace_back(1.0 / (static_cast<double>(mask_size) * static_cast<double>(mask_size)));
    }
    auto mask = TinyDIP::Image<double>(mask_data, mask_size, mask_size);
    auto output_image = TinyDIP::im2uint8(TinyDIP::conv2(TinyDIP::im2double(image1), mask));
    TinyDIP::bmp_write("OutputImages/1", output_image);
    TinyDIP::bmp_write("OutputImages/1_difference",
                        TinyDIP::im2uint8(TinyDIP::difference(TinyDIP::im2double(image1), TinyDIP::conv2(TinyDIP::im2double(image1), mask))
                        );

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}