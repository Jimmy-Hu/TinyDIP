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
    auto output = TinyDIP::paste2D(image1, image1, 100, 100);
    TinyDIP::bmp_write("OutputImages/paste2D", output);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}