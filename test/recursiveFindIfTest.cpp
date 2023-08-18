/* Developed by Jimmy Hu */

#include <execution>
#include <stdlib.h>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

void recursive_find_if_tests()
{
    auto test_vectors_1 = TinyDIP::n_dim_container_generator<4, int, std::vector>(1, 3);
    test_vectors_1[0][0][0][0] = 2;
    assert(TinyDIP::recursive_find_if<4>(test_vectors_1, [](auto&& i) { return i % 2 == 0; }));

    auto test_vectors_2 = TinyDIP::n_dim_container_generator<4, int, std::vector>(3, 3);
    assert(TinyDIP::recursive_find_if<4>(test_vectors_2, [](auto&& i) { return i % 2 == 0; }) == false);
    
    //  Tests with std::string
    auto test_vector_string = TinyDIP::n_dim_container_generator<4, std::string, std::vector>("1", 3);
    assert(TinyDIP::recursive_find_if<4>(test_vector_string, [](auto&& i) { return i == "1"; }));
    assert(TinyDIP::recursive_find_if<4>(test_vector_string, [](auto&& i) { return i == "2"; }) == false);

    //  Tests with std::string, projection
    assert(TinyDIP::recursive_find_if<4>(
        test_vector_string,
        [](auto&& i) { return i == "1"; },
        [](auto&& element) {return std::to_string(std::stoi(element) + 1); }) == false);
    

    return;
}

int main()
{
    auto start = std::chrono::system_clock::now();
    recursive_find_if_tests();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}