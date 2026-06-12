/* Developed by Jimmy Hu */

//  compile command:
//  clang++ -std=c++20 -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp  imageElementwiseAddTest.cpp -L /usr/local/Cellar/llvm/10.0.0_3/lib/ -lm -O3 -o imageElementwiseAddTest -v
//  https://stackoverflow.com/a/61821729/6667035
//  clear && rm -rf ./imageElementwiseAddTest && g++-11 -std=c++20 -O4 -ffast-math -funsafe-math-optimizations -std=c++20 -fpermissive -H --verbose -Wall imageElementwiseAddTest.cpp -o imageElementwiseAddTest 

#include <chrono>
#include <execution>
#include <map>
#include <omp.h>
#include <sstream>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"

void imageElementwiseAddTest(const std::size_t N1 = 10)
{
    auto test = TinyDIP::Image<int>(N1, 10);
    test.setAllValue(1);
    test += test;
    test.print();

    auto test2 = TinyDIP::Image<int>(N1 + 1, 11);
    test2.setAllValue(1);
    test += test2;
    test.print();
}

//  imageElementwiseWeightedAddTest Template Function Implementation
template<TinyDIP::arithmetic FloatingType = double>
void imageElementwiseWeightedAddTest(
    const std::filesystem::path& input_path1,
    const std::filesystem::path& input_path2,
    const std::string_view output_path,
    const FloatingType weight = 0.7
)
{
    if (!std::filesystem::exists(input_path1))
    {
        std::cerr << "File not found: " << input_path << '\n';
        return;
    }
    if (!std::filesystem::exists(input_path1))
    {
        std::cerr << "File not found: " << input_path << '\n';
        return;
    }
    TinyDIP::Image<TinyDIP::RGB> input_img1(0, 0);
    if (input_path1.extension() == ".bmp")
    {
        input_img1 = TinyDIP::bmp_read(input_path1.string().c_str(), true);
    }
    else
    {
        input_img1 = TinyDIP::pnm::read(std::execution::par, input_path1.string().c_str());
    }
    TinyDIP::Image<TinyDIP::RGB> input_img2(0, 0);
    if (input_path2.extension() == ".bmp")
    {
        input_img2 = TinyDIP::bmp_read(input_path2.string().c_str(), true);
    }
    else
    {
        input_img2 = TinyDIP::pnm::read(std::execution::par, input_path2.string().c_str());
    }
}

int main()
{
    TinyDIP::Timer timer1;
    imageElementwiseAddTest();
    return EXIT_SUCCESS;
}

