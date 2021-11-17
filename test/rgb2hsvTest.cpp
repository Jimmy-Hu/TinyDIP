#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void rgb2hsvTest();

int main()
{
    rgb2hsvTest();
	return 0;
}

void rgb2hsvTest()
{
    RGB rgb1{ 100, 100, 100 };
    auto hsv1 = TinyDIP::rgb2hsv(rgb1);
    std::cout << hsv1.channels[0] << ", " << hsv1.channels[1] << ", " << hsv1.channels[2] << '\n';
    auto rgb2 = TinyDIP::hsv2rgb(hsv1);
    std::cout << +rgb2.channels[0] << ", " << +rgb2.channels[1] << ", " << +rgb2.channels[2] << '\n';
}

