#include <chrono>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<typename ElementT>
void idct2Test(const std::size_t N1 = 10, const std::size_t N2 = 10)
{
	TinyDIP::Image<ElementT> test_input(N1, N2);
	for (std::size_t y = 1; y <= N2; y++)
	{
		for (std::size_t x = 1; x <= N1; x++)
		{
			test_input.at(y - 1, x - 1) = x * 10 + y;
		}
	}

	test_input.print();

	TinyDIP::idct2(TinyDIP::dct2(test_input)).print();

	return;
}

int main()
{
	auto start = std::chrono::system_clock::now();
	idct2Test<float>();
	idct2Test<double>();
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return 0;
}
