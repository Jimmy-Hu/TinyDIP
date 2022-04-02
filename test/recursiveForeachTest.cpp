#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void recursiveForeachTest();

int main()
{
	recursiveForeachTest();
	return 0;
}

void recursiveForeachTest()
{
	auto print = [](const int& n) { std::cout << " " << n; };
	std::vector<int> test_vector{ 1, 2, 3 };
	TinyDIP::recursive_for_each<1>(print, test_vector);
	std::cout << '\n';
	std::vector<decltype(test_vector)> test_vector2{ test_vector, test_vector, test_vector };
	TinyDIP::recursive_for_each<2>(test_vector2, print);

	return;
}