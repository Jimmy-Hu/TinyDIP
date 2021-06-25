/* Develop by Jimmy Hu */

//  compile command:
//  clang++ -std=c++20 -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp  bicubicInterpolationTest.cpp -L /usr/local/Cellar/llvm/10.0.0_3/lib/ -lm -O3 -o bicubicInterpolationTest -v
//  https://stackoverflow.com/a/61821729/6667035
//  clear && rm -rf ./bicubicInterpolationTest && g++-11 -std=c++20 -O4 -ffast-math -funsafe-math-optimizations -std=c++20 -fpermissive -H --verbose -Wall bicubicInterpolationTest.cpp -o bicubicInterpolationTest 

#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"

void bicubicInterpolationTest();

int main()
{
    bicubicInterpolationTest();
    return 0;
}

void bicubicInterpolationTest()
{
    TinyDIP::Image<GrayScale> image1(3, 3, 1);
    std::cout << "Width: " + std::to_string(image1.getSizeX()) + "\n";
    std::cout << "Height: " + std::to_string(image1.getSizeY()) + "\n";
    image1 = image1.set(1, 1, 100);
    image1.print();

    auto image2 = TinyDIP::copyResizeBicubic(image1, 12, 12);
    image2.print();
}
