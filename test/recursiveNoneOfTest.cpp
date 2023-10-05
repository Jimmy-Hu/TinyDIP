/* Developed by Jimmy Hu */

#include <execution>
#include <stdlib.h>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

void recursive_none_of_tests()
{
    

    return;
}

int main()
{
    auto start = std::chrono::system_clock::now();
    recursive_none_of_tests();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}