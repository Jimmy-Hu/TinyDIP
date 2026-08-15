#include <cassert>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

//  powTest Function Implementation
void powTest(const TinyDIP::RGB& input)
{
    std::size_t N1 = 10, N2 = 10;
    TinyDIP::Image<TinyDIP::RGB> test_input(N1, N2);
    test_input.setAllValue(input);
    TinyDIP::pow(test_input, 2.0).print(" ");
    return;
}

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
    TinyDIP::Timer timer1;
    powTest<int>();
    powTest<float>();
    powTest<double>();
    powTest<TinyDIP::RGB>(TinyDIP::RGB{2, 2, 2});
    return EXIT_SUCCESS;
}