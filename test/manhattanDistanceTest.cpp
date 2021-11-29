#include <cassert>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<class T>
void manhattanDistanceTest()
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
	T expected = 0;
	auto actual = TinyDIP::manhattan_distance(test_input, test_input);
	assert(actual == expected);

	auto test_input2 = test_input;
	test_input2.at(1, 1) = test_input2.at(1, 1) + 1;
	expected = 1;
	actual = TinyDIP::manhattan_distance(test_input, test_input2);
	std::string message = "expected: " + std::to_string(expected) + ",\tactual:" + std::to_string(actual) + '\n';
	std::cout << message;
	assert(actual == expected);
	return;
}

int main()
{
	manhattanDistanceTest<int>();
	manhattanDistanceTest<long>();
	manhattanDistanceTest<float>();
	manhattanDistanceTest<double>();
	return 0;
}
