#include <cassert>
#include <execution>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<class T>
void multipliesParallelTest(std::size_t start, std::size_t end)
{
	#ifdef USE_BOOST_ITERATOR
	for (std::size_t i = start; i <= end; i++)
	{
		std::size_t N1 = i, N2 = i;
		TinyDIP::Image<T> test_input(N1, N2);
		for (std::size_t y = 1; y <= N2; y++)
		{
			for (std::size_t x = 1; x <= N1; x++)
			{
				test_input.at(y - 1, x - 1) = x * 10 + y;
			}
		}
		auto expected = TinyDIP::multiplies(test_input, test_input);
		auto actual = TinyDIP::multiplies(std::execution::par, test_input, test_input);
		assert(actual == expected);
	}
	#endif
}

int main()
{
	multipliesParallelTest<int>(1, 20);
	multipliesParallelTest<long>(1, 20);
	multipliesParallelTest<float>(1, 20);
	multipliesParallelTest<double>(1, 20);
}