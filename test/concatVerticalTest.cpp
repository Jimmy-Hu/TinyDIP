#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void concatVerticalTest();

int main()
{
    auto start = std::chrono::system_clock::now();
	concatVerticalTest();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
	return 0;
}

void concatVerticalTest()
{
    TinyDIP::Image<GrayScale> image1(3, 3);
    image1.setAllValue(10);
    image1.at(2, 2) = 1;
    std::cout << "Width: " + std::to_string(image1.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image1.getHeight()) + "\n";
    auto image2 = TinyDIP::copyResizeBicubic(image1, 12, 12);
    std::cout << "Width: " + std::to_string(image2.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image2.getHeight()) + "\n";
    image2.print();
    image2 = TinyDIP::subimage(image2, 3, 3, 1, 1);
    image2.print();
    std::vector<decltype(image2)> v{ image2, image2 };
    image2 = TinyDIP::concat_vertical(v);
    //image2 = TinyDIP::concat_vertical(image2, image2);
    image2.print();
    return;
}

