//  Developed by Jimmy Hu

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

void splitOverlapTest(const std::filesystem::path& file_path, std::ostream& os = std::cout)
{
    auto input_image = TinyDIP::bmp_read(file_path.string().c_str(), true);
    auto split_overlap_output = TinyDIP::split_overlap(
        std::execution::seq,
        input_image,
        22,
        8,
        41,
        45
    );
    os << "split_overlap function execution finished.\n";
    for (std::size_t y = 0; y < std::ranges::size(split_overlap_output); ++y)
    {
        for (std::size_t x = 0; x < std::ranges::size(split_overlap_output[0]); ++x)
        {
            TinyDIP::bmp_write(std::to_string(x) + "_" + std::to_string(y), split_overlap_output[y][x]);
        }
    }
    
    return;
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << "<input_bmp>\n";
        return EXIT_SUCCESS;
    }
    std::filesystem::path input_path = std::string(argv[1]);
    if (!std::filesystem::exists(input_path))
    {
        std::cerr << "File not found: " << input_path << "\n";
        return EXIT_SUCCESS;
    }

    splitOverlapTest(input_path);
    return EXIT_SUCCESS;
}

