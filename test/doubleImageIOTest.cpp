#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

void doubleImageIOTest();

int main()
{
    auto start = std::chrono::system_clock::now();
    doubleImageIOTest();
    return 0;
    auto end = std::chrono::system_clock::now();
}

void doubleImageIOTest()
{
    std::size_t N1 = 10, N2 = 10;
    TinyDIP::Image<double> test_input(N1, N2);
    for (std::size_t y = 1; y <= N2; y++)
    {
        for (std::size_t x = 1; x <= N1; x++)
        {
            test_input.at(y - 1, x - 1) = x * 10 + y;
        }
    }

    test_input.print();

    auto dct_result = TinyDIP::dct2(test_input);
    dct_result.print();
    TinyDIP::double_image::write("test", dct_result);
    
    auto content_from_file = TinyDIP::double_image::read("test", false);
    content_from_file.print();

    std::cout << "Difference:\n";
    TinyDIP::subtract(dct_result, content_from_file).print();

    return;
}

