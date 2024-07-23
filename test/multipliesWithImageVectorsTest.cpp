#include <chrono>
#include <execution>
#include <sstream>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<class ElementT = double>
void multipliesWithImageVectorsTest(const std::size_t xsize, const std::size_t ysize, const std::size_t zsize)
{
    std::vector<TinyDIP::Image<ElementT>> test_x;
    for (std::size_t z = 0; z < zsize; ++z)
    {
        auto image = TinyDIP::Image<ElementT>(xsize, ysize);
        image.setAllValue(0.1);
        test_x.push_back(image);
    }
    std::vector<TinyDIP::Image<ElementT>> test_y;
    for (std::size_t z = 0; z < zsize; ++z)
    {
        auto image = TinyDIP::Image<ElementT>(xsize, ysize);
        image.setAllValue(0.2);
        test_y.push_back(image);
    }
    auto result = TinyDIP::pixelwise_multiplies(test_x, test_y);
    result.at(0).print();
    return;
}

template<class ExPo, class ElementT = double>
requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
void multipliesWithImageVectorsTest(ExPo execution_policy, const std::size_t xsize, const std::size_t ysize, const std::size_t zsize)
{
    std::vector<TinyDIP::Image<ElementT>> test_x;
    for (std::size_t z = 0; z < zsize; ++z)
    {
        auto image = TinyDIP::Image<ElementT>(xsize, ysize);
        image.setAllValue(0.2);
        test_x.push_back(image);
    }
    std::vector<TinyDIP::Image<ElementT>> test_y;
    for (std::size_t z = 0; z < zsize; ++z)
    {
        auto image = TinyDIP::Image<ElementT>(xsize, ysize);
        image.setAllValue(0.4);
        test_y.push_back(image);
    }
    auto result = TinyDIP::pixelwise_multiplies(execution_policy, test_x, test_y);
    result.at(0).print();
    return;
}

int main()
{
    auto start = std::chrono::system_clock::now();
    multipliesWithImageVectorsTest(10, 10, 10);
    //multipliesWithImageVectorsTest(std::execution::par, 5, 5, 5);    //    execution_policy unsupported for TinyDIP::Image vectors
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}