#include <cassert>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

int main()
{
    auto start = std::chrono::system_clock::now();
    static_assert(TinyDIP::get_from_variadic_template<1>(1, 2, 3, 4, 5) == 1);
    static_assert(TinyDIP::get_from_variadic_template<2>(1, 2, 3, 4, 5) == 2);
    static_assert(TinyDIP::get_from_variadic_template<3>(1, 2, 3, 4, 5) == 3);
    static_assert(TinyDIP::get_from_variadic_template<4>(1, 2, 3, 4, 5) == 4);
    static_assert(TinyDIP::get_from_variadic_template<5>(1, 2, 3, 4, 5) == 5);

    int A = 1;
    int B = 2;
    int C = 3;
    int D = 4;
    int E = 5;
    
    assert(TinyDIP::get_from_variadic_template<1>(A, B, C, D, E) == A);
    assert(TinyDIP::get_from_variadic_template<2>(A, B, C, D, E) == B);
    assert(TinyDIP::get_from_variadic_template<3>(A, B, C, D, E) == C);
    assert(TinyDIP::get_from_variadic_template<4>(A, B, C, D, E) == D);
    assert(TinyDIP::get_from_variadic_template<5>(A, B, C, D, E) == E);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return EXIT_SUCCESS;
}