#include <cassert>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<class T>
void powTest()
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
	TinyDIP::pow(test_input, 2).print();
	return;
}

int main()
{
	powTest<int>();
	powTest<float>();
	powTest<double>();
	return 0;
}