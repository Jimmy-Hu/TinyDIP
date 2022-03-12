/* Developed by Jimmy Hu */

//  compile command:
//  clang++ -std=c++20 -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp  bicubicInterpolationTest.cpp -L /usr/local/Cellar/llvm/10.0.0_3/lib/ -lm -O3 -o bicubicInterpolationTest -v
//  https://stackoverflow.com/a/61821729/6667035
//  clear && rm -rf ./bicubicInterpolationTest && g++-11 -std=c++20 -O4 -ffast-math -funsafe-math-optimizations -std=c++20 -fpermissive -H --verbose -Wall bicubicInterpolationTest.cpp -o bicubicInterpolationTest 

#include <chrono>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void bicubicInterpolationTest(const std::size_t sizex = 3, const std::size_t sizey = 3)
{
    TinyDIP::Image<GrayScale> image1(sizex, sizey, 1);
    std::cout << "Width: " + std::to_string(image1.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image1.getHeight()) + "\n";
    image1.at(1, 1) = 100;
    image1.print();

    auto image2 = TinyDIP::copyResizeBicubic(image1, 12, 12);
    image2.print();
}

int main()
{
    auto start = std::chrono::system_clock::now();
    bicubicInterpolationTest();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return 0;
}

