#include <cassert>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

int main()
{
    int A = 1;
    int B = 2;
    int C = 3;
    int D = 4;
    int E = 5;
    std::cout << TinyDIP::get_from_variadic_template<5>(A, B, C, D, E);
    static_assert(TinyDIP::get_from_variadic_template<1>(1, 2, 3, 4, 5) == 1);
    static_assert(TinyDIP::get_from_variadic_template<2>(1, 2, 3, 4, 5) == 2);
    static_assert(TinyDIP::get_from_variadic_template<3>(1, 2, 3, 4, 5) == 3);
    static_assert(TinyDIP::get_from_variadic_template<4>(1, 2, 3, 4, 5) == 4);
    static_assert(TinyDIP::get_from_variadic_template<5>(1, 2, 3, 4, 5) == 5);
    return 0;
}