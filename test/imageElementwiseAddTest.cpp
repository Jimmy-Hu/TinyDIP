/* Developed by Jimmy Hu */

//  compile command:
//  clang++ -std=c++20 -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp  imageElementwiseAddTest.cpp -L /usr/local/Cellar/llvm/10.0.0_3/lib/ -lm -O3 -o imageElementwiseAddTest -v
//  https://stackoverflow.com/a/61821729/6667035
//  clear && rm -rf ./imageElementwiseAddTest && g++-11 -std=c++20 -O4 -ffast-math -funsafe-math-optimizations -std=c++20 -fpermissive -H --verbose -Wall imageElementwiseAddTest.cpp -o imageElementwiseAddTest 

#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void imageElementwiseAddTest(const std::size_t N1 = 10)
{
    auto test = TinyDIP::Image<int>(N1, 10);
    test.setAllValue(1);
    test += test;
    test.print();

    auto test2 = TinyDIP::Image<int>(N1 + 1, 11, 1);
    test += test2;
    test.print();
}

int main()
{
    imageElementwiseAddTest();
    return 0;
}

