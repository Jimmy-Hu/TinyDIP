#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void concatTest();

int main()
{
    concatTest();
    return 0;
}

void concatTest()
{
    TinyDIP::Image<GrayScale> image1(3, 3, 10);
    image1.at(2, 2) = 1;
    auto image2 = TinyDIP::copyResizeBicubic(image1, 12, 12);
    image2 = TinyDIP::subimage(image2, 3, 3, 1, 1);
    std::vector<decltype(image2)> v1{ image2, image2 };
    std::vector<decltype(v1)> v2{ v1, v1 };

    TinyDIP::concat(v2).print();
    return;
}
