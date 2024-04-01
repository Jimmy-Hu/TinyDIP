/* Developed by Jimmy Hu */

#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
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

void recursiveReduceTest();

int main()
{
    auto start = std::chrono::system_clock::now();
    recursiveReduceTest(5, 5);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return 0;
}


void recursiveReduceTest(const std::size_t xsize, const std::size_t ysize)
{
    //  recursive_reduce test 1
    auto test_image1 = TinyDIP::Image<GrayScale>(xsize, ysize);
    test_image1.setAllValue(1);
    std::vector<decltype(test_image1)> test_vector_1{test_image1, test_image1, test_image1};
    auto expected_result_1 = TinyDIP::Image<GrayScale>(5, 5);
    expected_result_1.setAllValue(4);
    M_Assert(
        TinyDIP::recursive_reduce(test_vector_1, test_image1) ==
        expected_result_1,
        "recursive_reduce test 1 failed");
    std::cout << "All tests passed!\n";
    return;
}
