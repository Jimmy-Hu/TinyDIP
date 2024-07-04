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
    auto image1 = TinyDIP::Image<double>(3, 3);
    for (std::size_t y = 0; y < image1.getHeight(); ++y)
    {
        for (std::size_t x = 0; x < image1.getWidth(); ++x)
        {
            image1.at_without_boundary_check(x, y)
                             = static_cast<double>(x) + static_cast<double>(y) + 1;
        }
    }
    image1.print();
    TinyDIP::conv2(image1, image1).print();
    
    auto image2 = TinyDIP::bmp_read("InputImages/1", false);
    std::vector<unsigned char> mask_data = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    auto mask = TinyDIP::Image<unsigned char>(mask_data, std::size_t{3}, std::size_t{3});
    image2 = conv2(image2, mask);
    TinyDIP::bmp_write("OutputImages/1", image2);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}


