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
	return 0;
}

void isIntegerTest()
{

	return;
}