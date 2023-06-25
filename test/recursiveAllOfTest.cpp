/* Developed by Jimmy Hu */

#include <execution>
#include <stdlib.h>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

void recursiveAllOfTest();

int main()
{
    auto start = std::chrono::system_clock::now();
	recursiveAllOfTest();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return EXIT_SUCCESS;
}

void recursiveAllOfTest()
{
    auto test_vectors_1 = TinyDIP::n_dim_container_generator<4, int>(1, 3);
    test_vectors_1[0][0][0][0] = 2;
    std::cout << "Play with test_vectors_1:\n";
    
    if (TinyDIP::recursive_all_of(test_vectors_1, [](int i) { return i % 2 == 0; }))
        std::cout << "All numbers are even\n";

    auto test_vectors_2 = TinyDIP::n_dim_container_generator<4, int>(2, 3);
    test_vectors_2[0][0][0][0] = 4;
    std::cout << "Play with test_vectors_2:\n";
    
    if (TinyDIP::recursive_all_of(test_vectors_2, [](int i) { return i % 2 == 0; }))
        std::cout << "All numbers are even\n";
    return;
}