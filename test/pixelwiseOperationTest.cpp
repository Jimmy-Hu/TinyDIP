#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void pixelwiseOperationTest();

int main()
{
	pixelwiseOperationTest();
	return 0;
}

void pixelwiseOperationTest()
{
    constexpr std::size_t size = 10;
    
    auto img1 = TinyDIP::gaussianFigure2D(size, size, 5, 5, static_cast<double>(3));
	auto img2 = TinyDIP::gaussianFigure2D(size, size, 5, 5, static_cast<double>(3));
	auto img3 = TinyDIP::gaussianFigure2D(size, size, 5, 5, static_cast<double>(3));
	auto img4 = TinyDIP::gaussianFigure2D(size, size, 5, 5, static_cast<double>(3));

    auto output = TinyDIP::pixelwiseOperation
    (
        [](auto&& pixel_in_img1, auto&& pixel_in_img2, auto&& pixel_in_img3, auto&& pixel_in_img4)
        {
            return 2 * pixel_in_img1 + pixel_in_img2 - pixel_in_img3 * pixel_in_img4;
        },
        TinyDIP::pixelwiseOperation([](auto&& element) { return element; }, img1),
        TinyDIP::pixelwiseOperation([](auto&& element) { return element; }, img2),
        TinyDIP::pixelwiseOperation([](auto&& element) { return element; }, img3),
        TinyDIP::pixelwiseOperation([](auto&& element) { return element; }, img4)
    );
    output.print();
}


