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

//  backgroundTest function implementation
void backgroundTest(const std::filesystem::path& background_image_path, const std::filesystem::path& file_path, std::string_view output_path)
{
    auto background_image = TinyDIP::bmp_read(background_image_path.string().c_str(), true);
    auto background_image_1920x720 = TinyDIP::copyResizeBicubic(background_image, 1920, 720);
    if (!std::filesystem::exists(output_path))
    {
        auto image_input = TinyDIP::bmp_read(file_path.string().c_str(), true);
        TinyDIP::Image<TinyDIP::RGB> output_image(1920, 720);
        output_image = TinyDIP::pixelwise_transform([&](const auto& background_image_pixel, const auto& foreground_image_pixel)
        {
            if ((foreground_image_pixel.channels[0] == 0) && (foreground_image_pixel.channels[1] == 0) && (foreground_image_pixel.channels[2] == 0))
            {
                return background_image_pixel;
            }
            return foreground_image_pixel;
        }, background_image_1920x720, image_input);
        TinyDIP::bmp_write(std::string(output_path).c_str(), output_image);
    }
    return;
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    if(argc == 4)
    {
        std::filesystem::path input_background_image_path = std::string(argv[1]);
        if (!std::filesystem::exists(input_background_image_path))
        {
            throw std::runtime_error(TinyDIP::Formatter() << "File not found: " << input_background_image_path << '\n');
        }
        std::filesystem::path input_path = std::string(argv[2]);
        std::filesystem::path output_path = std::string(argv[3]);
        
        std::string target_ext = ".bmp";
        if (std::filesystem::is_directory(input_path) && std::filesystem::is_directory(output_path))
        {
            for (const auto & entry : std::filesystem::directory_iterator(input_path))
            {
                if (entry.is_regular_file() && entry.path().extension() == target_ext)
                {
                    std::cout << "Processing " << entry.path() << '\n';
                    backgroundTest(input_background_image_path, entry.path(), std::string(output_path / entry.path().stem()));
                }
            }
        }
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " <input background image> <input image folder> <output image folder>\n";
    }
    return EXIT_SUCCESS;
}