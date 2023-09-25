/* Developed by Jimmy Hu */

#include <execution>
#include <stdlib.h>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

void recursive_any_of_tests()
{
    auto test_vectors_1 = TinyDIP::n_dim_container_generator<4, int, std::vector>(1, 3);
    test_vectors_1[0][0][0][0] = 2;
    assert(TinyDIP::recursive_any_of<4>(test_vectors_1, [](auto&& i) { return i % 2 == 0; }));

    auto test_vectors_2 = TinyDIP::n_dim_container_generator<4, int, std::vector>(3, 3);
    assert(TinyDIP::recursive_any_of<4>(test_vectors_2, [](auto&& i) { return i % 2 == 0; }) == false);
    
    
    return;
}