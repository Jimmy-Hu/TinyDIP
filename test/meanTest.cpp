/* Developed by Jimmy Hu */

#include <execution>
#include <stdlib.h>
#include <thread>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

//  Copy from https://stackoverflow.com/a/37264642/6667035
#ifndef NDEBUG
#   define M_Assert(Expr, Msg) \
    M_Assert_Helper(#Expr, Expr, __FILE__, __LINE__, Msg)
#else
#   define M_Assert(Expr, Msg) ;
#endif

void M_Assert_Helper(const char* expr_str, bool expr, const char* file, int line, const char* msg)
{
    if (!expr)
    {
        std::cerr << "Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

//  mean_test template function implementation
template<class T, class ElementT = double>
void mean_test(const std::size_t sizex, const std::size_t sizey, ElementT initial_value = 1.5)
{
    auto test_image = TinyDIP::Image<T>(sizex, sizey);
    assert(TinyDIP::sum(test_image) == 0);
    test_image.setAllValue(initial_value);
    auto sum_result = std::reduce(std::ranges::cbegin(test_image.getImageData()), std::ranges::cend(test_image.getImageData()), 0, std::plus());
    sum_result /= test_image.count();
    if(TinyDIP::mean(test_image) != sum_result)
    {
        M_Assert(false, "Error occurred while calculating summation");
    }
    std::cout << TinyDIP::mean(test_image) << '\n';
    /*
    //  test with parallel execution policy
    if(TinyDIP::mean(std::execution::par, test_image) != sum_result)
    {
        M_Assert(false, "Error occurred while calculating summation");
    }
    */
    return;
}

void mean_test_double(const std::size_t sizex, const std::size_t sizey)
{
    mean_test<double>(sizex, sizey);
    return;
}

int main()
{
    auto start = std::chrono::system_clock::now();
    std::vector<std::thread> threads;
    //  Reference: https://stackoverflow.com/a/54551447/6667035
    for (size_t sizey = 0; sizey < 1000; ++sizey)
    {
        threads.emplace_back([&](){mean_test<double>(10, sizey);});
    }
    for (std::thread& each_thread : threads)
    {
        each_thread.join();
    }
    mean_test_double(10, 10);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}