#include <chrono>
#include <execution>
#include <sstream>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

template<class ElementT = double>
void multipliesWithImageVectorsTest(const std::size_t xsize, const std::size_t ysize, const std::size_t zsize)
{
	std::vector<TinyDIP::Image<ElementT>> test_x;
	for (std::size_t z = 0; z < zsize; ++z)
	{
		test_x.push_back(TinyDIP::Image(xsize, ysize, 0.1));
	}
	std::vector<TinyDIP::Image<ElementT>> test_y;
	for (std::size_t z = 0; z < zsize; ++z)
	{
		test_y.push_back(TinyDIP::Image(xsize, ysize, 0.2));
	}
	auto result = TinyDIP::multiplies(test_x, test_y);
	result.at(0).print();
	return;
}

int main()
{
	auto start = std::chrono::system_clock::now();
	multipliesWithImageVectorsTest(10, 10, 10);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return 0;
}