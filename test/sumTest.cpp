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
    __M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)
#else
#   define M_Assert(Expr, Msg) ;
#endif

void __M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg)
{
    if (!expr)
    {
        std::cerr << "Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

//  sum_test template function implementation
template<class T>
void sum_test(const std::size_t sizex, const std::size_t sizey)
{
    auto test_image = TinyDIP::Image<T>(sizex, sizey);
    assert(TinyDIP::sum(test_image) == 0);
    test_image.setAllValue(1);
    auto sum_result = std::reduce(std::ranges::cbegin(test_image.getImageData()), std::ranges::cend(test_image.getImageData()), 0, std::plus());
    if(TinyDIP::sum(test_image) != sum_result)
    {
        M_Assert(false, "Error occurred while calculating summation");
    }
    //  test with parallel execution policy
    if(TinyDIP::sum(std::execution::par, test_image) != sum_result)
    {
        M_Assert(false, "Error occurred while calculating summation");
    }
    return;
}

void sum_test_double(const std::size_t sizex, const std::size_t sizey)
{
    sum_test<double>(sizex, sizey);
    return;
}

int main()
{
    auto start = std::chrono::system_clock::now();
    std::thread t(sum_test_double, 10, 10);
    t.join();  
    sum_test<double>(10, 10);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}