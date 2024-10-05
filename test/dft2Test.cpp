/* Developed by Jimmy Hu */

#include <execution>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

int main()
{
    auto start = std::chrono::system_clock::now();
    TinyDIP::Image<double> I1(5, 5);
    for(std::size_t y = 0; y < I1.getHeight(); ++y)
    {
        for(std::size_t x = 0; x < I1.getWidth(); ++x)
        {
            I1.at(x, y) = y * I1.getWidth() + x + 1;
        }
    }
    I1.print();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}