#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<typename T>
void subimage2Test()
{
    TinyDIP::Image<T> image1(3, 3, 10);
    image1.at(2, 2) = 1;
    std::cout << "Width: " + std::to_string(image1.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image1.getHeight()) + "\n";
    auto image2 = TinyDIP::copyResizeBicubic(image1, 12, 12);
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
