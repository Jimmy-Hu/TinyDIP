/* Developed by Jimmy Hu */

#include <chrono>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void randFunctionTest(
    const std::size_t sizex = 3,
    const std::size_t sizey = 2)
{
    std::complex<double> complex1{ 1, 1 };
    std::cout << std::sin(complex1) << "\n\n";

    TinyDIP::RGB_DOUBLE rgb_double{ 4, -4, -4 };
    auto ep = std::execution::seq;
    std::cout << TinyDIP::isnan(
                 TinyDIP::abs(ep,
                 TinyDIP::acos(ep,
                 TinyDIP::atan(ep, rgb_double)))) << "\n\n";
    
    auto RGBDOUBLEimage1 = TinyDIP::constructRGBDOUBLE(TinyDIP::rand(sizex, sizey), TinyDIP::rand(sizex, sizey), TinyDIP::rand(sizex, sizey));
    std::cout << TinyDIP::cot(TinyDIP::to_complex(RGBDOUBLEimage1)) << "\n\n";
    auto RGBDOUBLEimage2 = TinyDIP::constructRGBDOUBLE(TinyDIP::rand(sizex, sizey), TinyDIP::rand(sizex, sizey), TinyDIP::rand(sizex, sizey));
    std::cout << "euclidean_distance of RGBDOUBLEimage1 and RGBDOUBLEimage2: " << '\n';
    std::cout << TinyDIP::euclidean_distance(RGBDOUBLEimage1, RGBDOUBLEimage2) << "\n";

    auto MultiChannelImage1 =
        TinyDIP::constructMultiChannel(
            TinyDIP::multiplies(TinyDIP::ones<std::complex<double>>(sizex, sizey), std::complex<double>{ 1.0, 1.0 }),
            TinyDIP::multiplies(TinyDIP::ones<std::complex<double>>(sizex, sizey), std::complex<double>{ 2.0, 2.0 }),
            TinyDIP::multiplies(TinyDIP::ones<std::complex<double>>(sizex, sizey), std::complex<double>{ 3.0, 3.0 }),
            TinyDIP::multiplies(TinyDIP::ones<std::complex<double>>(sizex, sizey), std::complex<double>{ 4.0, 4.0 }),
            TinyDIP::multiplies(TinyDIP::ones<std::complex<double>>(sizex, sizey), std::complex<double>{ 5.0, 5.0 }),
            TinyDIP::multiplies(TinyDIP::ones<std::complex<double>>(sizex, sizey), std::complex<double>{ 6.0, 6.0 }),
            TinyDIP::multiplies(TinyDIP::ones<std::complex<double>>(sizex, sizey), std::complex<double>{ 7.0, 7.0 }),
            TinyDIP::multiplies(TinyDIP::ones<std::complex<double>>(sizex, sizey), std::complex<double>{ 8.0, 8.0 })
        );
    auto MultiChannelImage2 = TinyDIP::Image<TinyDIP::MultiChannel<std::complex<double>, 8>>(sizex, sizey);
    std::cout << "euclidean_distance of MultiChannelImage1 and MultiChannelImage2: " << '\n';
    std::cout << TinyDIP::euclidean_distance(MultiChannelImage1, MultiChannelImage2) << "\n";

    using namespace std::complex_literals;
    std::cout << "abs function with TinyDIP::MultiChannel\n";
    TinyDIP::Image<TinyDIP::RGB> test_image(3);
    std::complex<long double> c = 1.0 + 1i;
    test_image.setAllValue(TinyDIP::RGB{ 1, 1, 1 });
    test_image.print(" ");

    //auto color_image = TinyDIP::constructRGBDOUBLE(TinyDIP::rand(sizex, sizey), TinyDIP::rand(sizex, sizey), TinyDIP::rand(sizex, sizey));
    //TinyDIP::to_complex(color_image).print(" ");
    return;
    //TinyDIP::recursive_print(TinyDIP::euclidean_distance(image1, image2))
    //std::cout << TinyDIP::euclidean_distance(TinyDIP::to_complex(image1), TinyDIP::to_complex(image2), std::complex(0.0, 0.0)) << '\n';
    
}

int main()
{
    auto start = std::chrono::system_clock::now();
    randFunctionTest();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    if (elapsed_seconds.count() != 1)
    {
        std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << " seconds.\n";
    }
    else
    {
        std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << " second.\n";
    }
    return EXIT_SUCCESS;
}
