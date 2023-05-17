#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../cube.h"
#include "../cube_operations.h"

template<class T>
constexpr void plusTest(const std::size_t N1 = 10, const std::size_t N2 = 10)
{
    auto image1 = TinyDIP::Image<T>(N1, N2);
	image1.setAllValue(1);
    auto vector1 = std::vector<decltype(image1)>();
    vector1.push_back(image1);
    TinyDIP::plus(vector1, vector1, vector1)[0].print();
}

int main()
{
	auto start = std::chrono::system_clock::now();
	plusTest<int>();
	plusTest<float>();
	plusTest<double>();
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return 0;
}