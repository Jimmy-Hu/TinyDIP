/* Developed by Jimmy Hu */

#include <cassert>
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

void recursive_remove_copy_if_tests()
{
    //  std::vector<int> test case
    std::vector<int> test_vector_1 = {
        1, 2, 3, 4, 5, 6
    };
    std::vector<int> expected_result_1 = {
        1, 3, 5
    };
    M_Assert(
        TinyDIP::recursive_remove_copy_if<1>(test_vector_1, [](auto&& x) { return (x % 2) == 0; }) ==
        expected_result_1,
        "std::vector<int> test case failed");
}