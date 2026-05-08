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
    if (argc == 2)
    {
        std::filesystem::path source_filename = std::string(argv[1]);
        std::cout << "Read image: " << source_filename.string() << '\n';
        auto input_image = TinyDIP::bmp_read(source_filename.string().c_str(), true);
        TinyDIP::double_image::write_to_csv(
            std::execution::par,
            source_filename.stem().string() + std::string("_output.csv"),
            TinyDIP::im2double(TinyDIP::getRplane(input_image))
        );
    }
    return EXIT_SUCCESS;
}