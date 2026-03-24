#include <algorithm>
#include <cassert>
#include <execution>
#include <filesystem>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../image_io.h"
#include "../cube.h"
#include "../cube_operations.h"
#include "../timer.h"

template<class T>
void manhattanDistanceTest()
{
    std::size_t N1 = 10, N2 = 10;
    TinyDIP::Image<T> test_input(N1, N2);
    for (std::size_t y = 1; y <= N2; y++)
    {
        for (std::size_t x = 1; x <= N1; x++)
        {
            test_input.at(y - 1, x - 1) = x * 10 + y + y & 1;
        }
    }
    T expected = 0;
    auto actual = TinyDIP::manhattan_distance(test_input, test_input);
    assert(actual == expected);

    auto test_input2 = test_input;
    test_input2.at(static_cast<std::size_t>(1), static_cast<std::size_t>(1)) = test_input2.at(static_cast<std::size_t>(1), static_cast<std::size_t>(1)) + 1;
    expected = 1;
    actual = TinyDIP::manhattan_distance(test_input, test_input2);
    std::string message = "expected: " + std::to_string(expected) + ",\tactual:" + std::to_string(actual) + '\n';
    std::cout << message;
    assert(actual == expected);
    return;
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    std::cout << "argc = " << std::to_string(argc) << '\n';
    if (argc < 3)
    {
        manhattanDistanceTest<int>();
        manhattanDistanceTest<long>();
        manhattanDistanceTest<float>();
        manhattanDistanceTest<double>();
    }
    else if (argc == 3)
    {
        std::filesystem::path input_path1 = std::string(argv[1]);
        std::filesystem::path input_path2 = std::string(argv[2]);
        if (!std::filesystem::exists(input_path1))
        {
            std::cerr << "File not found: " << input_path1 << '\n';
            return EXIT_SUCCESS;
        }
        if (!std::filesystem::exists(input_path2))
        {
            std::cerr << "File not found: " << input_path2 << '\n';
            return EXIT_SUCCESS;
        }
        auto input_image1 = TinyDIP::bmp_read(input_path1.string().c_str(), true);
        auto input_image2 = TinyDIP::bmp_read(input_path2.string().c_str(), true);
        auto manhattan_distance_output =  TinyDIP::manhattan_distance(input_image1, input_image2);
        std::cout << "R plane: " << +manhattan_distance_output[0] << '\n';
        std::cout << "G plane: " << +manhattan_distance_output[1] << '\n';
        std::cout << "B plane: " << +manhattan_distance_output[2] << '\n';
    }
    
    return EXIT_SUCCESS;
}
