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

int main()
{
    auto start = std::chrono::system_clock::now();
    transposeTest<int>();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}