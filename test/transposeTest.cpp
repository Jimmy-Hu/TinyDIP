#include <cassert>
#include <chrono>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"

//  remove_extension Function Implementation
//  Copy from: https://stackoverflow.com/a/6417908/6667035
std::string remove_extension(const std::string& filename)
{
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

//  transposeTest template function implementation
template<class T>
void transposeTest()
{
    std::size_t N1 = 2, N2 = 3;
    TinyDIP::Image<T> test_input(N1, N2);
    for (std::size_t y = 0; y < N2; ++y)
    {
        for (std::size_t x = 0; x < N1; ++x)
        {
            test_input.at(x, y) = x * 10 + y;
        }
    }

    test_input.print();
    TinyDIP::transpose(test_input).print();
}

void transposeTestWithFile(const std::string& image_filename)
{
    auto input_img = TinyDIP::bmp_read(image_filename.c_str(), true);
    if (input_img.getDimensionality() != 2)
    {
        std::cerr << "Input image is not a 2D image!\n";
        return;
    }
    auto transpose_result = TinyDIP::transpose(input_img);
    for (std::size_t test_loop = 0; test_loop < 10; test_loop++)
    {
        std::chrono::high_resolution_clock::time_point start_time1 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < 100; ++i)
        {
            transpose_result = TinyDIP::transpose(transpose_result);
        }
        std::chrono::duration<double> elapsed_seconds1 = std::chrono::high_resolution_clock::now() - start_time1;
        std::print(std::cout, "Without execution policy: elapsed time: {} seconds.\n", elapsed_seconds1.count());
        std::chrono::high_resolution_clock::time_point start_time2 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < 100; ++i)
        {
            transpose_result = TinyDIP::transpose(std::execution::par, transpose_result);
        }
        std::chrono::duration<double> elapsed_seconds2 = std::chrono::high_resolution_clock::now() - start_time2;
        std::print(std::cout, "With execution policy: elapsed time: {} seconds.\n", elapsed_seconds2.count());
    }
    TinyDIP::bmp_write(
        "../OutputImages/RainImages/S__55246868_transpose",
        transpose_result);
}

int main()
{
    TinyDIP::Timer timer;
    transposeTest<int>();
    std::string input_path = "../InputImages/RainImages/S__55246868.bmp";
    if (!std::filesystem::is_regular_file(input_path))
    {
        input_path = "InputImages/RainImages/S__55246868.bmp";
    }
    transposeTestWithFile(input_path);
    return EXIT_SUCCESS;
}