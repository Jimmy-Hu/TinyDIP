/* Developed by Jimmy Hu */
#include <chrono>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<typename T>
void absTest(T initVal)
{
    std::size_t sizeNum = 10;
    auto test_img = TinyDIP::Image<T>(sizeNum, sizeNum);
    test_img.setAllValue(initVal);
    TinyDIP::abs(test_img).print();
    return;
}

template<typename T = TinyDIP::RGB_DOUBLE>
void absRGB_DOUBLETest(T initVal)
{
    std::size_t sizeNum = 10;
    auto test_img = TinyDIP::Image<T>(sizeNum, sizeNum);
    test_img.setAllValue(initVal);
    TinyDIP::abs(test_img).print();
    TinyDIP::abs(std::execution::seq, test_img).print();
}

int main()
{
    auto start = std::chrono::system_clock::now();
    absTest<int>(-10);
    absTest<float>(-10.0);
    absTest<double>(-10.0);
    absRGB_DOUBLETest(TinyDIP::RGB_DOUBLE{ -1.0, -2.0, -3.0 });
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    if (elapsed_seconds.count() != 1)
    {
        std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << " seconds.\n";
    }
    else
    {
        std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << " second.\n";
    }
    return EXIT_SUCCESS;
}
