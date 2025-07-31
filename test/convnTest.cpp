/* Developed by Jimmy Hu */
//  Online Test: https://godbolt.org/z/77qTv3KhG


#include <cassert>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"

int main()
{
    TinyDIP::Timer timer1;
    auto image1 = TinyDIP::ones<double>(std::size_t{ 2 }, std::size_t{ 3 }, std::size_t{ 2 }, std::size_t{ 2 });
    TinyDIP::convolution(image1, image1).print();
    return EXIT_SUCCESS;
}