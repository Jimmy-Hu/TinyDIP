/* Developed by Jimmy Hu */

#include <execution>
#include <stdlib.h>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

template<std::size_t dim>
void allOfTest();

int main()
{
    auto start = std::chrono::system_clock::now();
    allOfTest<2>();
    allOfTest<3>();
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

template<std::size_t dim>
void allOfTest()
{
    if constexpr(dim == 2)
    {
        auto test_image1 = TinyDIP::Image<double>(10, 10);
        test_image1.setAllValue(10);
        assert(TinyDIP::all_of(test_image1, [](int i) { return i == 10; }));
    }
    if constexpr(dim == 3)
    {
        auto test_image1 = TinyDIP::Image<double>(10, 10, 10);
        test_image1.setAllValue(10);
        assert(TinyDIP::all_of(test_image1, [](int i) { return i == 10; }));
    }
    return;
}