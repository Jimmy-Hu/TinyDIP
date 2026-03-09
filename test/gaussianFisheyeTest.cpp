//  Developed by Jimmy Hu

#include <cassert>
#include <execution>
#include <filesystem>
#include <string>
#include <string_view>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../image_io.h"
#include "../cube.h"
#include "../cube_operations.h"
#include "../timer.h"

template<std::floating_point FloatingType = double>
void gaussianFisheyeTest(std::string_view input_filename, std::string_view output_filename, FloatingType D0)
{
    std::filesystem::path source_path = std::string(input_filename);
    if (!std::filesystem::exists(source_path))
    {
        throw std::runtime_error(TinyDIP::Formatter() << "File = " << input_filename << " not found!");
    }
    auto input_image = TinyDIP::bmp_read(std::string(input_filename).c_str(), true);
    std::filesystem::path destination_path = std::string(output_filename);
    std::filesystem::path path_without_extension = destination_path.parent_path() / destination_path.stem();
    TinyDIP::bmp_write(path_without_extension.string(), TinyDIP::gaussian_fisheye(input_image, D0));
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <*.bmp> <*.bmp> <D0>\n";
        return EXIT_FAILURE;
    }
    std::string source_filename(argv[1]);
    std::string destination_filename(argv[2]);
    auto D0_input = std::stod(argv[3]);
    gaussianFisheyeTest(source_filename, destination_filename, D0_input);
    return EXIT_SUCCESS;
}