/* Developed by Jimmy Hu */

#include <execution>
#include <cstdlib>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"

//  From https://stackoverflow.com/a/37264642/6667035
#ifndef NDEBUG
#   define M_Assert(Expr, Msg) \
    M_Assert_Helper(#Expr, Expr, __FILE__, __LINE__, Msg)
#else
#   define M_Assert(Expr, Msg) ;
#endif

void M_Assert_Helper(const char* expr_str, bool expr, const char* file, int line, std::string msg)
{
    if (!expr)
    {
        std::cerr << "Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

void recursive_minmax_tests()
{
    auto test_vector = TinyDIP::n_dim_container_generator<3>(3, 3);
    test_vector.at(0).at(0).at(0) = 5;
    test_vector.at(0).at(0).at(1) = -5;
    test_vector.at(0).at(0).at(2) = -6;

    auto [min_number, max_number] = TinyDIP::recursive_minmax<3>(test_vector);
    M_Assert(
        max_number == 5,
        "recursive_minmax test case failed");
    M_Assert(
        min_number == -6,
        "recursive_minmax test case failed");
    return;
}

int main()
{
    TinyDIP::Timer timer1;
    recursive_minmax_tests();
    return EXIT_SUCCESS;
}