#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void pixelwise_transformParallelTest();

int main()
{
	pixelwise_transformParallelTest();
	return 0;
}

void pixelwise_transformParallelTest()
{
	for (size_t test_size = 0; test_size < 100; test_size++)
    {
		auto img1 = TinyDIP::gaussianFigure2D(test_size, test_size, 5, 5, static_cast<double>(3));
        
        auto output = TinyDIP::pixelwise_transform
        (
            std::execution::par,
            [](auto&& pixel_in_img1)
            {
                return 2 * pixel_in_img1;
            },
            TinyDIP::pixelwise_transform([](auto&& element) { return element; }, img1)
        );
        std::cout << TinyDIP::recursive_reduce(TinyDIP::subtract(output, TinyDIP::pixelwise_transform
        (
            [](auto&& pixel_in_img1)
            {
                return 2 * pixel_in_img1;
            },
            TinyDIP::pixelwise_transform([](auto&& element) { return element; }, img1)
        )).getImageData(), 0) << '\n';
    }
	return;
}