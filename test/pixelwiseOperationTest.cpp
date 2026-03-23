#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<typename T>
void pixelwise_transformTest(const size_t size = 10)
{
    auto img1 = TinyDIP::gaussianFigure2D(size, size, 5, 5, static_cast<T>(3));
	auto img2 = TinyDIP::gaussianFigure2D(size, size, 5, 5, static_cast<T>(3));
	auto img3 = TinyDIP::gaussianFigure2D(size, size, 5, 5, static_cast<T>(3));
	auto img4 = TinyDIP::gaussianFigure2D(size, size, 5, 5, static_cast<T>(3));

    auto output = TinyDIP::pixelwise_transform
    (
        [](auto&& pixel_in_img1, auto&& pixel_in_img2, auto&& pixel_in_img3, auto&& pixel_in_img4)
        {
            return 2 * pixel_in_img1 + pixel_in_img2 - pixel_in_img3 * pixel_in_img4;
        },
        TinyDIP::pixelwise_transform([](auto&& element) { return element; }, img1),
        TinyDIP::pixelwise_transform([](auto&& element) { return element; }, img2),
        TinyDIP::pixelwise_transform([](auto&& element) { return element; }, img3),
        TinyDIP::pixelwise_transform([](auto&& element) { return element; }, img4)
    );
    output.print();
}

int main()
{
    auto start = std::chrono::system_clock::now();
    pixelwise_transformTest<int>();
    pixelwise_transformTest<long>();
    pixelwise_transformTest<float>();
	pixelwise_transformTest<double>();
    auto end = std::chrono::system_clock::now();
	return 0;
}


