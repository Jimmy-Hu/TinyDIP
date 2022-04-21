#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void recursiveTransformTest();

template<typename ElementT>
void print3(std::vector<TinyDIP::Image<ElementT>> input)
{
	for (std::size_t i = 0; i < input.size(); i++)
	{
		input[i].print();
		std::cout << "*******************\n";
	}
}

int main()
{
	auto start = std::chrono::system_clock::now();
	recursiveTransformTest();
	auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return 0;
}

void recursiveTransformTest()
{
	for (std::size_t N = 1; N < 10; ++N)
	{
		std::size_t N1 = N, N2 = N, N3 = N;
		auto test_vector = TinyDIP::n_dim_vector_generator<3>(0, 10);

		for (std::size_t z = 1; z <= N3; z++)
		{
			for (std::size_t y = 1; y <= N2; y++)
			{
				for (std::size_t x = 1; x <= N1; x++)
				{
					test_vector.at(z - 1).at(y - 1).at(x - 1) = x * 100 + y * 10 + z;
				}
			}
		}
		auto expected = TinyDIP::recursive_transform<3>([](auto&& element) {return element + 1; }, test_vector);
		auto actual = TinyDIP::recursive_transform<3>(std::execution::par, [](auto&& element) {return element + 1; }, test_vector);
		std::cout << "N = " << N << ": " << std::to_string(actual == expected) << '\n';
	}
	
	//print3(TinyDIP::recursive_transform<1>([](std::vector<std::vector<int>> element) { return TinyDIP::Image<int>(element); }, test_vector));
}