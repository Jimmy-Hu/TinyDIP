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
    std::vector<double> mask_data = {0.11111111111111111111111111111111,
                                     0.11111111111111111111111111111111,
                                     0.11111111111111111111111111111111,
                                     0.11111111111111111111111111111111,
                                     0.11111111111111111111111111111111,
                                     0.11111111111111111111111111111111,
                                     0.11111111111111111111111111111111,
                                     0.11111111111111111111111111111111,
                                     0.11111111111111111111111111111111};
    auto mask = TinyDIP::Image<double>(mask_data, std::size_t{3}, std::size_t{3});
    auto output_image = conv2(image1.cast<double>(), mask).cast<unsigned char>();
    TinyDIP::bmp_write("OutputImages/1", output_image);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}