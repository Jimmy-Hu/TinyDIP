#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../cube.h"
#include "../cube_operations.h"

template<class T>
constexpr void plusTest(const std::size_t N1 = 10, const std::size_t N2 = 10)
{
    auto image1 = TinyDIP::Image<T>(N1, N2, 1);
    auto vector1 = std::vector<TinyDIP::Image<double>>();
    vector1.push_back(image1);
    TinyDIP::plus(vector1, vector1, vector1)[0].print();
}

