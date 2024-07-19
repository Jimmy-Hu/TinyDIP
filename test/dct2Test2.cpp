/* Developed by Jimmy Hu */

#include <execution>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

void each_image( std::string input_path, std::string output_path,
                 std::size_t N1 = 8, std::size_t N2 = 8)
{
    auto input_img = TinyDIP::bmp_read(input_path.c_str(), false);
    auto image_255 = TinyDIP::Image<double>(input_img.getWidth(), input_img.getHeight());
    image_255.setAllValue(255);
    auto dct2_results = TinyDIP::recursive_transform<2>(
        std::execution::seq,
        [](auto&& element) { return TinyDIP::dct2(element); },
        TinyDIP::split(
            TinyDIP::divides(
                TinyDIP::getVplane(
                    TinyDIP::rgb2hsv(input_img)),
                    image_255),
            input_img.getWidth() / N1,
            input_img.getHeight() / N2)
        );
    auto dct2_combined = TinyDIP::concat(dct2_results);
    TinyDIP::double_image::write(output_path.c_str(), dct2_combined);
}

void dct2Test2( std::string arg1, std::string arg2,
                std::string arg3,
                std::size_t N1 = 8, std::size_t N2 = 8)
{
    std::cout << "dct2Test2 program..." << '\n';
    std::cout << arg1 << '\n';
    std::cout << arg2 << '\n';
    std::size_t start_index = 50, end_index = 100;
    for (std::size_t i = start_index; i <= end_index; i++)
    {
        std::string fullpath = arg1 + "/" + std::to_string(i);
        std::cout << fullpath << '\n';
        auto output_path = arg3 + "/" + std::to_string(i);
        each_image(fullpath, output_path, N1, N2);
    }
    auto output_path = arg3 + "/GT";
    each_image(arg2, output_path, N1, N2);
    return;
}

int main(int argc, char* argv[])
{
    auto start = std::chrono::system_clock::now();
    if(argc == 3)
    {
        auto arg1 = std::string(argv[1]);
        auto arg2 = std::string(argv[2]);
        auto arg3 = std::string(argv[3]);
        dct2Test2(arg1, arg2, arg3);
    }
    else
    {
        
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}

