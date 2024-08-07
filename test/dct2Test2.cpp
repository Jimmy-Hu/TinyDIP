/* Developed by Jimmy Hu 
    dct2Test2.cpp is an example for converting spatial domain image to DCT domain image
    each_image Function takes input_path of BMP file and output_path of DBMP (double BMP) file

*/

#include <execution>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

//  each_image Function Implementation
void each_image( std::string input_path, std::string output_path,
                 std::size_t N1 = 8, std::size_t N2 = 8)
{
    auto input_img = TinyDIP::bmp_read(input_path.c_str(), false);

    auto mod_x = std::fmod(static_cast<double>(input_img.getWidth()), static_cast<double>(N1));
    auto mod_y = std::fmod(static_cast<double>(input_img.getHeight()), static_cast<double>(N2));

    input_img = TinyDIP::subimage(
        input_img,
        input_img.getWidth() - mod_x, input_img.getHeight() - mod_y,
        static_cast<double>(input_img.getWidth()) / 2.0, static_cast<double>(input_img.getHeight()) / 2.0
    );
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

//  dct2_split_divides_255 Template Function Implementation
template<class ElementT>
constexpr static auto dct2_split_divides_255(
    const TinyDIP::Image<ElementT>& input,
    std::size_t N1 = 8,
    std::size_t N2 = 8)
{
    auto image_255 = TinyDIP::Image<double>(input.getWidth(), input.getHeight());
    image_255.setAllValue(255);
    return TinyDIP::recursive_transform<2>(
        std::execution::seq,
        [](auto&& element) { return TinyDIP::dct2(element); },
        TinyDIP::split(
            TinyDIP::divides(
                input,
                image_255),
            input.getWidth() / N1,
            input.getHeight() / N2)
    );
}

//  each_image_SuperResolution Function Implementation
/*
* each_image_SuperResolution Function performs block based DCT on each channel of a RGB image
*/
void each_image_SuperResolution(
    std::string input_path,
    std::string output_path,
    std::size_t N1 = 8,
    std::size_t N2 = 8)
{
    auto input_img = TinyDIP::bmp_read(input_path.c_str(), false);

    auto mod_x = std::fmod(static_cast<double>(input_img.getWidth()), static_cast<double>(N1));
    auto mod_y = std::fmod(static_cast<double>(input_img.getHeight()), static_cast<double>(N2));

    input_img = TinyDIP::subimage(
        input_img,
        input_img.getWidth() - mod_x, input_img.getHeight() - mod_y,
        static_cast<double>(input_img.getWidth()) / 2.0, static_cast<double>(input_img.getHeight()) / 2.0
    );
    auto dct2_R_combined = 
        TinyDIP::concat(dct2_split_divides_255(TinyDIP::im2double(TinyDIP::getRplane(input_img)), N1, N2));
    TinyDIP::double_image::write((output_path + std::string("_R")).c_str(), dct2_R_combined);
    auto dct2_G_combined = 
        TinyDIP::concat(dct2_split_divides_255(TinyDIP::im2double(TinyDIP::getGplane(input_img)), N1, N2));
    TinyDIP::double_image::write((output_path + std::string("_G")).c_str(), dct2_G_combined);
    auto dct2_B_combined = 
        TinyDIP::concat(dct2_split_divides_255(TinyDIP::im2double(TinyDIP::getBplane(input_img)), N1, N2));
    TinyDIP::double_image::write((output_path + std::string("_B")).c_str(), dct2_B_combined);
}

//  imageSuperResolutionExperiment Function Implementation
void imageSuperResolutionExperiment(
    std::string input_high_res_img_path = "InputImages/HighRes",
    std::string output_high_res_img_path = "Dictionary/HighRes",
    std::string input_low_res_img_path = "InputImages/LowRes/Bucubic0.1",
    std::string output_low_res_img_path = "Dictionary/LowRes/Bucubic0.1",
    std::size_t high_res_N1 = 80,
    std::size_t high_res_N2 = 80,
    std::size_t low_res_N1 = 8,
    std::size_t low_res_N2 = 8)
{
    std::cout << "imageSuperResolutionExperiment function...\n";
    std::size_t n_zero = 4;
    std::size_t start_index = 1, end_index = 10;
    #pragma omp parallel for
    for (std::size_t i = start_index; i <= end_index; i++)
    {
        std::string old_str = std::to_string(i);
        std::string fullpath = input_high_res_img_path + std::string("/") + std::string(n_zero - std::min(n_zero, old_str.length()), '0') + old_str;;
        std::cout << fullpath << '\n';
        auto output_path = output_high_res_img_path + std::string("/") + std::to_string(i);
        each_image_SuperResolution(fullpath, output_path, high_res_N1, high_res_N2);
    }
    #pragma omp parallel for
    for (std::size_t i = start_index; i <= end_index; i++)
    {
        std::string old_str = std::to_string(i);
        std::string fullpath = input_low_res_img_path + std::string("/") + std::string(n_zero - std::min(n_zero, old_str.length()), '0') + old_str;;
        std::cout << fullpath << '\n';
        auto output_path = output_low_res_img_path + std::string("/") + std::to_string(i);
        each_image_SuperResolution(fullpath, output_path, low_res_N1, low_res_N2);
    }
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
        imageSuperResolutionExperiment();
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << " seconds. \n";
    return EXIT_SUCCESS;
}

