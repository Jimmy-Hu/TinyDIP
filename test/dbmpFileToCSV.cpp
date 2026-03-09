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

void dbmpFileToCSV(std::string_view source, std::string_view destination, std::ostream& os = std::cout)
{
    os << "Determine the input filename " << source << " \n";
    std::filesystem::path source_path = std::string(source);
    if (!std::filesystem::exists(source_path))
    {
        throw std::runtime_error(TinyDIP::Formatter() << "File = " << source << " not found!");
    }
    auto input_dbmp = TinyDIP::double_image::read(std::string(source).c_str(), true);
    TinyDIP::double_image::write_to_csv(std::string(destination).c_str(), input_dbmp);
    return;
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <*.dbmp> <*.csv>\n";
        return EXIT_FAILURE;
    }
    std::string source_filename(argv[1]);
    std::string destination_filename(argv[2]);
    dbmpFileToCSV(source_filename, destination_filename);
    return EXIT_SUCCESS;
}