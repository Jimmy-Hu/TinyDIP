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

void bicubicInterpolationTest(
    const std::size_t sizex = 3,
    const std::size_t sizey = 3,
    const std::size_t init_val = 1,
    const std::size_t new_sizex = 12,
    const std::size_t new_sizey = 12)
{
    TinyDIP::Image<TinyDIP::GrayScale> image1(sizex, sizey);
    image1.setAllValue(init_val);
    std::cout << "Width: " + std::to_string(image1.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image1.getHeight()) + "\n";
    image1.at(static_cast<std::size_t>(1), static_cast<std::size_t>(1)) = 100;
    image1.print();

    auto image2 = TinyDIP::copyResizeBicubic(image1, new_sizex, new_sizey);
    std::cout << "Width after copyResizeBicubic operation: " + std::to_string(image2.getWidth()) + "\n";
    std::cout << "Height after copyResizeBicubic operation: " + std::to_string(image2.getHeight()) + "\n";
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
    return EXIT_SUCCESS;
}

