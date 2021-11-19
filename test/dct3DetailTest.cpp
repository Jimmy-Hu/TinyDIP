#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void dct3DetailTest();
void print3();

int main()
{
	dct3DetailTest();
	return 0;
}

template<typename ElementT>
void print3(std::vector<TinyDIP::Image<ElementT>> input)
{
	for (std::size_t i = 0; i < input.size(); i++)
	{
		input[i].print();
		std::cout << "*******************\n";
	}
}

void dct3DetailTest()
{
	std::size_t N1 = 10, N2 = 10, N3 = 10;
	std::vector<TinyDIP::Image<double>> test_input;
	for (std::size_t z = 0; z < N3; z++)
	{
		test_input.push_back(TinyDIP::Image<double>(N1, N2));
	}
	for (std::size_t z = 1; z <= N3; z++)
	{
		for (std::size_t y = 1; y <= N2; y++)
		{
			for (std::size_t x = 1; x <= N1; x++)
			{
				test_input[z - 1].at(y - 1, x - 1) = x * 100 + y * 10 + z;
			}
		}
	}
	print3(test_input);

	auto test_output = TinyDIP::dct3_detail(test_input, 0);
	test_output.print();

}