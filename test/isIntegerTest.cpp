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
	
	return 0;
}

void isIntegerTest()
{

	return;
}