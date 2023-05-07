/* Developed by Jimmy Hu */
#include <chrono>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<typename T>
void absTest(T initVal)
{
	std::size_t sizeNum = 10;
	auto test_img = TinyDIP::Image<T>(sizeNum, sizeNum);
	test_img.setAllValue(initVal);
	TinyDIP::abs(test_img).print();
	return;
}

int main()
{
	auto start = std::chrono::system_clock::now();
	absTest<int>(-10);
	absTest<float>(-10.0);
	absTest<double>(-10.0);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return 0;
}
