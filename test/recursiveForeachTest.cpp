#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void recursiveForeachTest();

int main()
{
	auto start = std::chrono::system_clock::now();
	recursiveForeachTest();
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return 0;
}

void recursiveForeachTest()
{
	//	Arrange
	auto print = [](const int& n) { std::cout << " " << n; };
	std::vector<int> test_vector{ 1, 2, 3 };

	//	Action
	TinyDIP::recursive_for_each<1>(print, test_vector);
	std::cout << '\n';
	std::vector<decltype(test_vector)> test_vector2{ test_vector, test_vector, test_vector };
	TinyDIP::recursive_for_each<2>(print, test_vector2);
	return;
}