/* Developed by Jimmy Hu */

#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../timer.h"

void recursiveSizeTest();

int main()
{
    recursiveSizeTest();
    return EXIT_SUCCESS;
}

void recursiveSizeTest()
{
    std::vector<std::vector<std::vector<int>>> test_v{
        {{1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {{1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {{1, 1}, {1, 1}, {1, 1}, {1, 1}} };
    constexpr std::size_t N = 3;
    std::cout << TinyDIP::recursive_size<N>(test_v);
}