#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void absTest();

int main()
{
	absTest();
	return 0;
}

void absTest()
{
	std::size_t sizeNum = 10;
	auto test_img = TinyDIP::Image(sizeNum, sizeNum, -10);
	TinyDIP::abs(test_img).print();
	return;
}