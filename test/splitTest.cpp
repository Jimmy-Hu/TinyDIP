#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void splitTest();

int main()
{
	splitTest();
	return 0;
}

void splitTest()
{
    TinyDIP::Image<GrayScale> image1(3, 3, 10);
    image1.at(2, 2) = 1;
    std::cout << "Width: " + std::to_string(image1.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image1.getHeight()) + "\n";
    auto image2 = TinyDIP::copyResizeBicubic(image1, 12, 12);
    std::cout << "Width: " + std::to_string(image2.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image2.getHeight()) + "\n";
    image2.print();
    auto test_output = TinyDIP::split(image2, 2, 2);
    TinyDIP::recursive_for_each<2>(test_output, [](TinyDIP::Image<GrayScale> element) { element.print(); });
	return;
}