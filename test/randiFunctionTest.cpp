/* Developed by Jimmy Hu */

#include <chrono>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../timer.h"

void randiFunctionTest(
    const std::size_t sizex = 3,
    const std::size_t sizey = 2)
{
    //  Zero-dimensional result, an `Image<ElementT>` is returned
    auto randi_output1 = TinyDIP::randi(10);
    std::cout << "Zero-dimensional result: " << randi_output1 << '\n';

    //  Zero-dimensional result with specified range
    auto randi_output2 = TinyDIP::randi(std::make_pair(10, 100));
    std::cout << "Zero-dimensional result with specified range: " << randi_output2 << '\n';

    //  One-dimensional result
    auto randi_output3 = TinyDIP::randi(10, sizex);
    std::cout << "One-dimensional result: \n";
    randi_output3.print();

    //  sizex-by-sizey image of pseudorandom integers
    auto randi_output4 = TinyDIP::randi(10, sizex, sizey);
    std::cout << "sizex-by-sizey image of pseudorandom integers: \n";
    randi_output4.print();

    //  One-dimensional result with specified range
    auto randi_output5 = TinyDIP::randi(std::make_pair(10, 100), sizex);
    std::cout << "One-dimensional result with specified range: \n";
    randi_output5.print();

    //  sizex-by-sizey image of pseudorandom integers with specified range
    auto randi_output6 = TinyDIP::randi(std::make_pair(10, 100), sizex, sizey);
    std::cout << "sizex-by-sizey image of pseudorandom integers with specified range: \n";
    randi_output6.print();

    return;
}

int main()
{
    TinyDIP::Timer timer1;
    randiFunctionTest();
    return EXIT_SUCCESS;
}
