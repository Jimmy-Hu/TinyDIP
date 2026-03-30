//  Developed by Jimmy Hu

#include <algorithm>
#include <cassert>
#include <execution>
#include <filesystem>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../image_io.h"
#include "../cube.h"
#include "../cube_operations.h"
#include "../timer.h"

void conv2Test3(
    const std::filesystem::path& input_path1,
    const std::filesystem::path& input_path2,
    const std::string_view output_path
)
{
    auto input_img1 = TinyDIP::bmp_read(input_path1.string().c_str(), true);
    auto input_img2 = TinyDIP::bmp_read(input_path2.string().c_str(), true);
    auto input_img1_r_double = TinyDIP::im2double(TinyDIP::getRplane(input_img1));
    auto input_img2_r_double = TinyDIP::im2double(TinyDIP::getRplane(input_img2));
    auto input_img1_g_double = TinyDIP::im2double(TinyDIP::getGplane(input_img1));
    auto input_img2_g_double = TinyDIP::im2double(TinyDIP::getGplane(input_img2));
    auto input_img1_b_double = TinyDIP::im2double(TinyDIP::getBplane(input_img1));
    auto input_img2_b_double = TinyDIP::im2double(TinyDIP::getBplane(input_img2));
    auto output_image_r = TinyDIP::conv2(input_img1_r_double, input_img2_r_double, true);
    auto output_image_g = TinyDIP::conv2(input_img1_g_double, input_img2_g_double, true);
    auto output_image_b = TinyDIP::conv2(input_img1_b_double, input_img2_b_double, true);
    if (true)           //  Save to .dbmp files
    {
        TinyDIP::double_image::write("output_image_r", output_image_r);
        TinyDIP::double_image::write("output_image_g", output_image_g);
        TinyDIP::double_image::write("output_image_b", output_image_b);
    }
    TinyDIP::double_image::write_to_csv("conv2Test3Test_R.csv", output_image_r);
    TinyDIP::bmp_write(std::string(output_path).c_str(), TinyDIP::im2uint8(TinyDIP::constructRGBDOUBLE(output_image_r, output_image_g, output_image_b)));
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    std::cout << "argc = " << std::to_string(argc) << '\n';
    if(argc == 3)
    {
        std::filesystem::path input_path1 = std::string(argv[1]);
        if (!std::filesystem::exists(input_path1))
        {
            std::cerr << "File not found: " << input_path1 << '\n';
            return EXIT_SUCCESS;
        }
        std::filesystem::path input_path2 = std::string(argv[2]);
        if (!std::filesystem::exists(input_path2))
        {
            std::cerr << "File not found: " << input_path2 << '\n';
            return EXIT_SUCCESS;
        }
        conv2Test3(input_path1, input_path2, std::string("convolution2Output"));
    }
    else if  (argc == 4)
    {
        std::filesystem::path input_path1 = std::string(argv[1]);
        if (!std::filesystem::exists(input_path1))
        {
            std::cerr << "File not found: " << input_path1 << '\n';
            return EXIT_SUCCESS;
        }
        std::filesystem::path input_path2 = std::string(argv[2]);
        if (!std::filesystem::exists(input_path2))
        {
            std::cerr << "File not found: " << input_path2 << '\n';
            return EXIT_SUCCESS;
        }
        std::filesystem::path output_path = std::string(argv[3]);
        std::filesystem::path path_without_extension = output_path.parent_path() / output_path.stem();
        conv2Test3(input_path1, input_path2, path_without_extension.string());
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " <input_image_path1> <input_image_path1> [output_image_path]\n";
    }
    return EXIT_SUCCESS;
}