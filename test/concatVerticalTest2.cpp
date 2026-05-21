//  Developed by Jimmy Hu
//  Input 2 images (A, B) which sizes are the same, the output image is top side of A + bottom side of B


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

//  concatVerticalTest2 Template Function Implementation
template<
    class ExecutionPolicy
>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
static auto concatVerticalTest2(
    ExecutionPolicy&& policy,
    const std::filesystem::path& input_path1,
    const std::filesystem::path& input_path2,
    const std::filesystem::path& output_path = "concatVerticalTest2Output"
)
{
    TinyDIP::Image<TinyDIP::RGB> input_image1(0, 0);
    if (input_path1.extension() == ".bmp")
    {
        input_image1 = TinyDIP::bmp_read(input_path1.string().c_str(), true);
    }
    else
    {
        input_image1 = TinyDIP::pnm::read(std::forward<ExecutionPolicy>(policy), input_path1.string().c_str());
    }
    TinyDIP::Image<TinyDIP::RGB> input_image2(0, 0);
    if (input_path2.extension() == ".bmp")
    {
        input_image2 = TinyDIP::bmp_read(input_path2.string().c_str(), true);
    }
    else
    {
        input_image2 = TinyDIP::pnm::read(std::forward<ExecutionPolicy>(policy), input_path2.string().c_str());
    }
    if (input_image1.getSize() != input_image2.getSize())
    {
        throw std::runtime_error("Size mismatched!");
    }
    
    auto center_location = static_cast<std::size_t>(static_cast<double>(input_image1.getSize(1)) / 2.0);
    auto top_side_image = TinyDIP::subimage2(input_image1, 0, input_image1.getSize(0) - 1, 0, center_location - 1);
    auto bottom_side_image = TinyDIP::subimage2(input_image2, 0, input_image2.getSize(0) - 1, center_location, input_image2.getSize(1) - 1);
    auto output_image = TinyDIP::concat_vertical(top_side_image, bottom_side_image);
    TinyDIP::bmp_write(output_path.string().c_str(), output_image);
    return;
}


int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    std::cout << "argc = " << std::to_string(argc) << '\n';
    if(argc < 3)
    {
        std::cout << "Usage: " << std::string(argv[0]) << " <input_file1> <input_file2> [output_file]\n";
    }
    else if (argc == 3)
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
        concatVerticalTest2(std::execution::par_unseq, input_path1, input_path2);
    }
}
