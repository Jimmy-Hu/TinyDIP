#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void dct2Test();

int main()
{
	dct2Test();
	return 0;
}

void dct2Test()
{
	std::cout << "dct2Test program..." << '\n';
	std::size_t N1 = 10, N2 = 10;
	TinyDIP::Image<double> test_input(N1, N2);
	for (std::size_t y = 1; y <= N2; ++y)
	{
		for (std::size_t x = 1; x <= N1; ++x)
		{
			test_input.at(y - 1, x - 1) = x * 10 + y;
		}
	}

	test_input.print();

	TinyDIP::dct2(test_input).print();
}
