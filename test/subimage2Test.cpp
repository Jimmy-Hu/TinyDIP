#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<typename T>
void subimage2Test(const std::size_t sizex = 3, const std::size_t sizey = 3)
{
    TinyDIP::Image<T> image1(sizex, sizey, 10);
    image1.at(2, 2) = 1;
    std::cout << "Width: " + std::to_string(image1.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image1.getHeight()) + "\n";
    auto image2 = TinyDIP::copyResizeBicubic<T>(image1, 12, 12);
    std::cout << "Width: " + std::to_string(image2.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image2.getHeight()) + "\n";
    image2.print();
    image2 = TinyDIP::subimage2(image2, 0, 3, 0, 2);
    image2.print();
    return;
}

int main()
{
    subimage2Test<int>();
    subimage2Test<long>();
    subimage2Test<float>();
    subimage2Test<double>();
    return 0;
}
