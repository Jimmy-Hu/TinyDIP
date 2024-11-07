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
    return EXIT_SUCCESS;
}

void gaussianFigure2DTest()
{
    auto gaussian_image = TinyDIP::gaussianFigure2D(10, 10, 5, 5, 1.0, 0.0, 1.0);
    gaussian_image.print();
    return;
}