/* Developed by Jimmy Hu */

#include <algorithm>
#include <cstdlib>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../image_io.h"
#include "../cube.h"
#include "../cube_operations.h"
#include "../timer.h" 

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <input_image_csv>";
        return EXIT_SUCCESS;
    }
    else if(argc == 2)
    {
        std::filesystem::path source_filename = std::string(argv[1]);
        std::cout << "Read image: " << source_filename.string() << '\n';
        auto source_image = TinyDIP::double_image::read_from_csv(source_filename.string().c_str());
        for (std::size_t rotate_degree = 1; rotate_degree < 359; ++rotate_degree)
        {
            auto rotated_image = TinyDIP::rotate_detail_shear_transformation_degree(source_image, static_cast<long double>(rotate_degree));
            TinyDIP::double_image::write_to_csv(
                (source_filename.stem().string() + std::string("_") + std::to_string(rotate_degree) + std::string(".csv")).c_str(),
                rotated_image
            );
        }
        
    }
    else if(argc == 3)
    {
        std::filesystem::path source_filename = std::string(argv[1]);
        std::filesystem::path destination_filename = std::string(argv[2]);
        std::cout << "Read image: " << source_filename.string() << '\n';
        auto source_image = TinyDIP::double_image::read_from_csv(source_filename.string().c_str());
        
    }

    return EXIT_SUCCESS;
}