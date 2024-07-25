/* Developed by Jimmy Hu */
//  Online Test: https://godbolt.org/z/77qTv3KhG


#include <cassert>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

int main()
{
    auto start = std::chrono::system_clock::now();
    auto image1 = TinyDIP::ones<double>(std::size_t{2}, std::size_t{2}, std::size_t{2}, std::size_t{2});
    TinyDIP::convolution(image1, image1).print();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}