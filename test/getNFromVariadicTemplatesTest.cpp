#include <cassert>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

int main()
{
    static_assert(std::is_same<TinyDIP::get_from_variadic_template_t<1, int, long>, int>::value);
    static_assert(std::is_same<TinyDIP::get_from_variadic_template_t<2, int, long>, long>::value);
    return 0;
}