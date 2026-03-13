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

//  pnmFileReadTest function implementation
void pnmFileReadTest(const std::filesystem::path& file_path, std::string_view output_path)
{
    auto image_input = TinyDIP::pnm::read(file_path);
    TinyDIP::bmp_write(std::string(output_path).c_str(), image_input);
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << "<input_bmp> <output_bmp>\n";
        return EXIT_SUCCESS;
    }
    
    std::filesystem::path input_path = std::string(argv[1]);
    if (!std::filesystem::exists(input_path))
    {
        std::cerr << "File not found: " << input_path << "\n";
        return EXIT_SUCCESS;
    }
    std::filesystem::path output_path = std::string(argv[2]);
    std::filesystem::path path_without_extension = output_path.parent_path() / output_path.stem();
    pnmFileReadTest(input_path, path_without_extension.string());
    return EXIT_SUCCESS;
}