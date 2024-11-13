/* Developed by Jimmy Hu */

#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void gaussianFigure2DTest();

int main()
{
    auto start = std::chrono::system_clock::now();
    gaussianFigure2DTest();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    if (elapsed_seconds.count() != 1)
    {
    return EXIT_SUCCESS;
}

void gaussianFigure2DTest()
{
    auto gaussian_image = TinyDIP::gaussianFigure2D(10, 10, 5, 5, 1.0, 0.0, 1.0);
    gaussian_image.print();
    return;
}