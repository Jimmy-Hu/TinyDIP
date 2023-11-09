/* Developed by Jimmy Hu */

#include <execution>
#include <stdlib.h>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

template<class T>
void sum_test(int sizex, int sizey)
{
    auto test_image = TinyDIP::Image<T>(sizex, sizey);
    assert(TinyDIP::sum(test_image) == 0);
    test_image.setAllValue(1);
    auto sum_result = std::reduce(std::ranges::cbegin(test_image.getImageData()), std::ranges::cend(test_image.getImageData()), 0, std::plus())
    if(TinyDIP::sum(test_image) != sum_result)
    {
        assert(false, "Error occurred while calculating summation");
    }
    return;
}

int main()
{
    auto start = std::chrono::system_clock::now();
    sum_test<double>(10, 10);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}