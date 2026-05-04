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
        if (true)                   //  add white boundary
        {
            source_image = TinyDIP::add_border_2d(std::execution::par, source_image, 3, TinyDIP::max(source_image));
        }  
        for (std::size_t rotate_degree = 1; rotate_degree < 359; ++rotate_degree)
        {
            auto rotated_image = TinyDIP::rotate_detail_shear_transformation_degree(source_image, static_cast<long double>(rotate_degree));
            auto output_filename_csv = source_filename.stem().string() + std::string("_") + std::to_string(rotate_degree) + std::string(".csv");
            if (!std::filesystem::exists(output_filename_csv))
            {
                TinyDIP::double_image::write_to_csv(
                    output_filename_csv.c_str(),
                    rotated_image
                );
            }
            rotated_image = TinyDIP::multiplies(TinyDIP::normalize(rotated_image), 255.0);
            auto output_filename_bmp = source_filename.stem().string() + std::string("_") + std::to_string(rotate_degree);
            TinyDIP::bmp_write(output_filename_bmp, TinyDIP::constructRGB(
                TinyDIP::im2uint8(rotated_image),
                TinyDIP::im2uint8(rotated_image),
                TinyDIP::im2uint8(rotated_image)
            ));
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