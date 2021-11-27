#include <cassert>
#include <execution>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<class T>
void powParallelTest()
{
	for (std::size_t i = 0; i < 11; i++)
	{
		std::size_t N1 = 10, N2 = 10;
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
	powParallelTest<int>();
	powParallelTest<float>();
	powParallelTest<double>();
	return 0;
}