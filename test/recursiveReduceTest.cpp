#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void recursiveReduceTest();

int main()
{
    auto start = std::chrono::system_clock::now();
    recursiveReduceTest();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return 0;
}


void recursiveReduceTest()
{
    auto test_image1 = TinyDIP::Image<GrayScale>(5, 5);
    
}
