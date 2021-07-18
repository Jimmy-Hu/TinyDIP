/* Developed by Jimmy Hu */

//  compile command:
//  clang++ -std=c++20 -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp  main.cpp -L /usr/local/Cellar/llvm/10.0.0_3/lib/ -lm -O3 -o main -v
//  https://stackoverflow.com/a/61821729/6667035
//  clear && rm -rf ./main && g++-11 -std=c++20 -O4 -ffast-math -funsafe-math-optimizations -std=c++20 -fpermissive -H --verbose -Wall main.cpp -o main 

#include "basic_functions.h"

int main()
{
    bicubicInterpolationTest();
    
    auto bmp1 = TinyDIP::bmp_read("2", false);
    auto bmp2 = TinyDIP::bmp_read("DerainOutput5_Data2_frame23", false);
    bmp1 = TinyDIP::subtract(bmp1, bmp2);
    TinyDIP::bmp_write("test", bmp1);

    std::cout << "*********\n";
    auto img1 = TinyDIP::Image<GrayScale>(10, 10, 1);
    auto img2 = TinyDIP::Image<GrayScale>(10, 10, 2);
    auto img3 = TinyDIP::Image<GrayScale>(10, 10, 3);
    auto img4 = TinyDIP::Image<GrayScale>(10, 10, 4);
    TinyDIP::subtract(TinyDIP::plus(img1, img2, img3, img4), img4).print();
    return 0;
    
    
    
    #ifdef USE_BOOST_ITERATOR
    std::vector<int> a{1, 2, 3}, b{4, 5, 6};
    auto result = TinyDIP::recursive_transform<1>(
        std::execution::par,
        [](int element1, int element2) { return element1 * element2; },
        a, b);
    for (size_t i = 0; i < result.size(); i++)
    {
        std::cout << result.at(i) << std::endl;
    }

    std::vector<decltype(a)> c{a, a, a}, d{b, b, b};
    auto result2 = TinyDIP::recursive_transform<2>(
        std::execution::par,
        [](int element1, int element2) { return element1 * element2; },
        c, d);
    TinyDIP::recursive_print(result2);
    #endif
    
    return 0;
}

void test()
{
    constexpr int dims = 5;
    std::vector<std::string> test_vector1{ "1", "4", "7" };
    auto test1 = TinyDIP::n_dim_vector_generator<dims>(test_vector1, 3);
    std::vector<std::string> test_vector2{ "2", "5", "8" };
    auto test2 = TinyDIP::n_dim_vector_generator<dims>(test_vector2, 3);
    std::vector<std::string> test_vector3{ "3", "6", "9" };
    auto test3 = TinyDIP::n_dim_vector_generator<dims>(test_vector3, 3);
    std::vector<std::string> test_vector4{ "a", "b", "c" };
    auto test4 = TinyDIP::n_dim_vector_generator<dims>(test_vector4, 3);
    auto output = TinyDIP::recursive_transform<dims + 1>(
        [](auto element1, auto element2, auto element3, auto element4) { return element1 + element2 + element3 + element4; },
        test1, test2, test3, test4);
    std::cout << typeid(output).name() << std::endl;
    TinyDIP::recursive_print(output
    .at(0).at(0).at(0).at(0).at(0));
    return;   
}
