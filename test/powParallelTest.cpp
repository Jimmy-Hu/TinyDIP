#include <cassert>
#include <chrono>
#include <execution>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<class T>
void powParallelTest(const std::size_t N1 = 10)
{
	for (std::size_t i = 0; i < 11; i++)
	{
		std::size_t N2 = 10;
		TinyDIP::Image<T> test_input(N1, N2);
		for (std::size_t y = 1; y <= N2; y++)
		{
			for (std::size_t x = 1; x <= N1; x++)
			{
				test_input.at(y - 1, x - 1) = x * 10 + y;
			}
		}
		auto expected = TinyDIP::pow(test_input, 2);
		auto actual = TinyDIP::pow(std::execution::par, test_input, 2);
		assert(actual == expected);
	}
	return;
}

int main()
{
	auto start = std::chrono::system_clock::now();
	powParallelTest<int>();
	powParallelTest<float>();
	powParallelTest<double>();
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return 0;
}