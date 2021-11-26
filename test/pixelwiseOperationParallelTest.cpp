#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void pixelwiseOperationParallelTest();

int main()
{
	pixelwiseOperationParallelTest();
	return 0;
}

void pixelwiseOperationParallelTest()
{
	for (size_t test_size = 0; test_size < 100; test_size++)
    {
		auto img1 = TinyDIP::gaussianFigure2D(test_size, test_size, 5, 5, static_cast<double>(3));
        
        auto output = TinyDIP::pixelwiseOperation
        (
            std::execution::par,
            [](auto&& pixel_in_img1)
            {
                return 2 * pixel_in_img1;
            },
            TinyDIP::pixelwiseOperation([](auto&& element) { return element; }, img1)
        );
        std::cout << TinyDIP::recursive_reduce(TinyDIP::subtract(output, TinyDIP::pixelwiseOperation
        (
            [](auto&& pixel_in_img1)
            {
                return 2 * pixel_in_img1;
            },
            TinyDIP::pixelwiseOperation([](auto&& element) { return element; }, img1)
        )).getImageData(), 0) << '\n';
    }
	return;
}