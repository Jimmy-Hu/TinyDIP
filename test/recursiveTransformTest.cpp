/* Developed by Jimmy Hu */

#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

//  Copy from https://stackoverflow.com/a/37264642/6667035
#ifndef NDEBUG
#   define M_Assert(Expr, Msg) \
    __M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)
#else
#   define M_Assert(Expr, Msg) ;
#endif

void __M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg)
{
    if (!expr)
    {
        std::cerr << "Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

template<typename InputT>
void recursiveTransformTest(InputT initialValue)
{
	for (std::size_t N = 1; N < 10; ++N)
	{
		std::size_t N1 = N, N2 = N, N3 = N;
		auto test_vector = TinyDIP::n_dim_vector_generator<3>(initialValue, 10);

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
		M_Assert(
			expected == actual,
			"recursive_transform test failed");
	}
	
	//print3(TinyDIP::recursive_transform<1>([](std::vector<std::vector<int>> element) { return TinyDIP::Image<int>(element); }, test_vector));
}

void recursiveTransformArrayTest()
{
    std::array<int, 5> test_array_1{1, 2, 3, 4, 5};
    std::array<decltype(test_array_1), 2> test_array_2{test_array_1, test_array_1};
    auto result = TinyDIP::recursive_transform<2>(
		[&](auto&& element1, auto&& element2, auto&& element3)
		{
			return element1 + element2 + element3;
		},
		test_array_2,
		test_array_2,
		test_array_2);
	std::array<int, 5> expected_result_1{3, 6, 9, 12, 15};
    std::array<decltype(expected_result_1), 2> expected_result_2{expected_result_1, expected_result_1};
    assert(result == expected_result_2);
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

int main()
{
	auto start = std::chrono::system_clock::now();
	std::cout << "Test with int:\n";
	recursiveTransformTest(3);
	recursiveTransformTest(static_cast<double>(3));
	recursiveTransformTest(static_cast<float>(3));
	recursiveTransformArrayTest();
	auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return 0;
}

