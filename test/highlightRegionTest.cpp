#include <chrono>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<typename T>
void subimage2Test(const std::size_t sizex = 3, const std::size_t sizey = 3)
{
    TinyDIP::Image<T> image1(sizex, sizey);
    image1.setAllValue(10);
    image1.at(static_cast<std::size_t>(2), static_cast<std::size_t>(2)) = 1;
    std::cout << "Width: " + std::to_string(image1.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image1.getHeight()) + "\n";
    auto image2 = TinyDIP::copyResizeBicubic<T>(image1, 12, 12);
    std::cout << "Width: " + std::to_string(image2.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image2.getHeight()) + "\n";
    image2.print();
    image2 = TinyDIP::subimage2(image2, 0, 3, 0, 2);
    image2.print();
    return;
}

int main()
{
    auto start = std::chrono::system_clock::now();
    subimage2Test<int>();
    subimage2Test<long>();
    subimage2Test<float>();
    subimage2Test<double>();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return 0;
}
