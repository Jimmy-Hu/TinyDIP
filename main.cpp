/* Developed by Jimmy Hu */

//  compile command:
//  clang++ -std=c++20 -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp  main.cpp -L /usr/local/Cellar/llvm/10.0.0_3/lib/ -lm -O3 -o main -v
//  https://stackoverflow.com/a/61821729/6667035
//  clear && rm -rf ./main && g++-11 -std=c++20 -O4 -ffast-math -funsafe-math-optimizations -std=c++20 -fpermissive -H --verbose -Wall main.cpp -o main 

#include "basic_functions.h"

int main()
{
    std::vector<int> a{1, 2, 3}, b{4, 5, 6};
    /*
    std::cout << a.at(0) << std::endl;
    std::cout << a.at(1) << std::endl;
    std::cout << a.at(2) << std::endl;
    */
    auto result = TinyDIP::recursive_transform<1>(
        [](int element1, int element2) { return element1 * element2; },
        a, b);
    for (size_t i = 0; i < result.size(); i++)
    {
        std::cout << result.at(i) << std::endl;
    }

    std::vector<decltype(a)> c{a, a, a}, d{b, b, b};
    auto result2 = TinyDIP::recursive_transform<2>(
        [](int element1, int element2) { return element1 * element2; },
        c, d);
    TinyDIP::recursive_print(result2);
    
    return 0;
}
