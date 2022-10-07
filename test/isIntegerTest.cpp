#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void isIntegerTest();

int main()
{
	auto start = std::chrono::system_clock::now();
	isIntegerTest();
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return 0;
}

void isIntegerTest()
{
	assert(TinyDIP::is_integer(1) == true);
	assert(TinyDIP::is_integer(2) == true);
	assert(TinyDIP::is_integer(3) == true);
	assert(TinyDIP::is_integer(1.1) == false);

	float test_number1 = 1.2;
	assert(TinyDIP::is_integer(test_number1) == false);
	test_number1 = 1;
	assert(TinyDIP::is_integer(test_number1) == true);

	double test_number2 = 2;
	assert(TinyDIP::is_integer(test_number2) == true);

	return;
}