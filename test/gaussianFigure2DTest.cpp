/* Developed by Jimmy Hu */

#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"

void gaussianFigure2DTest();

int main()
{
    TinyDIP::Timer timer1;
    gaussianFigure2DTest();
    return EXIT_SUCCESS;
}

void gaussianFigure2DTest()
{
    auto gaussian_image = TinyDIP::gaussianFigure2D(10, 10, 5, 5, 1.0, 0.0, 1.0);
    gaussian_image.print();
    auto gaussian_plane =
        TinyDIP::gaussianFigure2D(
            1024,
            1024,
            512,
            512,
            500.0,
            500.0,
            0.7
        );
    gaussian_plane = TinyDIP::multiplies(TinyDIP::normalize(gaussian_plane), 255.0);
    TinyDIP::bmp_write("test_gaussian",
        TinyDIP::constructRGB(
            TinyDIP::im2uint8(gaussian_plane),
            TinyDIP::im2uint8(gaussian_plane),
            TinyDIP::im2uint8(gaussian_plane)
        )
    );
    return;
}