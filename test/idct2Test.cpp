#include <chrono>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void idct2Test();

int main()
{
	idct2Test();
	return 0;
}

void idct2Test()
{
	std::size_t N1 = 10, N2 = 10;
	TinyDIP::Image<double> test_input(N1, N2);
	for (std::size_t y = 1; y <= N2; y++)
	{
		for (std::size_t x = 1; x <= N1; x++)
		{
			test_input.at(y - 1, x - 1) = x * 10 + y;
		}
	}

	test_input.print();

	TinyDIP::idct2(TinyDIP::dct2(test_input)).print();

	return;
}