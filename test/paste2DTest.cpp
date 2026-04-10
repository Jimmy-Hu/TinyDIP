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

//  paste2DTest template function implementation
template<class ExecutionPolicy>
requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
void paste2DTest(ExecutionPolicy&& execution_policy, const std::filesystem::path& file_path, std::string_view output_path)
{
    if (!std::filesystem::exists(output_path))
    {
        auto image_input = TinyDIP::bmp_read(file_path.string().c_str(), true);
        image_input = TinyDIP::copyResizeBicubic(image_input, 1280, 720);
        TinyDIP::Image<TinyDIP::RGB> output_image(1920, 720);
        output_image = TinyDIP::paste2D(std::forward<ExecutionPolicy>(execution_policy), output_image, image_input, 0, 0);
        TinyDIP::bmp_write(std::string(output_path).c_str(), output_image);
    }
    return;
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
                    paste2DTest(std::execution::par, entry.path(), std::string(output_path / entry.path().stem()));
                }
            }
        }
    }
    else
    {
        auto image1 = TinyDIP::bmp_read("InputImages/1", false);
        auto output = TinyDIP::paste2D(image1, image1, 100, 100);
        TinyDIP::bmp_write("OutputImages/paste2D", output);
    }
    return EXIT_SUCCESS;
}