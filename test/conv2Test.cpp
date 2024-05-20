/* Developed by Jimmy Hu */

#include <cassert>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

int main()
{
    auto start = std::chrono::system_clock::now();
    auto image1 = TinyDIP::Image<double>(3, 3);
    for (std::size_t y = 0; y < image1.getHeight(); ++y)
    {
        for (std::size_t x = 0; x < image1.getWidth(); ++x)
        {
            image1.at(x, y) = static_cast<double>(x) + 1.0;
        }
    }
    TinyDIP::conv2(image1, image1).print();
    
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}


