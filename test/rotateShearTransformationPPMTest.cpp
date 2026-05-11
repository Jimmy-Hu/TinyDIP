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
        std::cout << "Usage: " << argv[0] << " <input_image_ppm>";
        return EXIT_SUCCESS;
    }
    else if(argc == 2)
    {
        std::filesystem::path source_filename = std::string(argv[1]);
        std::cout << "Read image: " << source_filename.string() << '\n';
        auto source_image = TinyDIP::pnm::read(std::execution::seq, source_filename);
        auto rotated_image = TinyDIP::rotate_detail_shear_transformation_degree(source_image, static_cast<long double>(90));
        rotated_image = TinyDIP::lanczos_resample(rotated_image, 1080, 1920);
        auto output_filename_ppm = source_filename.stem().string() + std::string("_") + std::to_string(90) + std::string(".ppm");
        if (!std::filesystem::exists(output_filename_ppm))
        {
            TinyDIP::pnm::write(
                rotated_image,
                output_filename_ppm.c_str()
            );
        }
        
    }
    else if(argc == 3)
    {
        std::filesystem::path source_filename = std::string(argv[1]);
        std::filesystem::path destination_filename = std::string(argv[2]);
        std::cout << "Read image: " << source_filename.string() << '\n';
        auto source_image = TinyDIP::pnm::read(std::execution::seq, source_filename);
        
    }

    return EXIT_SUCCESS;
}