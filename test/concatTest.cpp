#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void concatTest();

int main()
{
    auto start = std::chrono::system_clock::now();
    concatTest();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}

void concatTest()
{
    TinyDIP::Image<TinyDIP::GrayScale> image1(3, 3);
    image1.setAllValue(10);
    image1.at(static_cast<std::size_t>(2), static_cast<std::size_t>(2)) = 1;
    std::cout << "Width: " + std::to_string(image1.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image1.getHeight()) + "\n";
    auto image2 = TinyDIP::copyResizeBicubic(image1, 12, 12);
    std::cout << "Width: " + std::to_string(image2.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image2.getHeight()) + "\n";
    image2.print();
    auto test_output = TinyDIP::concat(TinyDIP::split(image2, 2, 2));
    test_output.print();
    return;
}
