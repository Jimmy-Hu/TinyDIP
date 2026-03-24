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

//  offsetTest Function Implementation
void offsetTest(const std::filesystem::path& file_path, std::string_view output_path)
{
    auto image_input = TinyDIP::bmp_read(file_path.string().c_str(), true);
    auto output_image = TinyDIP::pixelwise_transform(
            std::execution::par_unseq,
            [](auto&& element)
            {
                TinyDIP::RGB output;
                for(std::size_t channel_index = 0; channel_index < 3; ++channel_index)
                {
                    if (element.channels[channel_index] < 10)
                    {
                        output.channels[channel_index] = 64;
                    }
                    else
                    {
                        output.channels[channel_index] = std::clamp(static_cast<int>(static_cast<double>(element.channels[channel_index]) * 0.78), 0, 255);
                    }
                }
                return output;
            }, image_input);
    TinyDIP::bmp_write(std::string(output_path).c_str(), output_image);
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    if(argc == 3)
    {
        std::filesystem::path input_path = std::string(argv[1]);
        std::filesystem::path output_path = std::string(argv[2]);
        if (!std::filesystem::exists(input_path))
        {
            std::cerr << "File / Path not found: " << input_path << '\n';
            return EXIT_SUCCESS;
        }
        std::string target_ext = ".bmp";
        if (std::filesystem::is_directory(input_path) && std::filesystem::is_directory(output_path))
        {
            for (const auto & entry : std::filesystem::directory_iterator(input_path))
            {
                if (entry.is_regular_file() && entry.path().extension() == target_ext)
                {
                    std::cout << "Processing " << entry.path() << '\n';
                    offsetTest(entry.path(), std::string(output_path / entry.path().stem()));
                }
            }
        }
    }
    else
    {
        
    }
    return EXIT_SUCCESS;
}




