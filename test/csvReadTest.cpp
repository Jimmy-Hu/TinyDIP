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

void create_dummy_csv(const char* const filename)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << "1.1,2.2,3.3\n";
        file << "4.4,5.5,6.6\n";
        file << "7.7,8.8,9.9\n";
    }
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    if (argc < 2)
    {
        const char* const test_filename = "test_matrix.csv";
        
        create_dummy_csv(test_filename);
        
        std::cout << "Testing OpenMP Fallback Read:\n";
        auto img_omp = TinyDIP::double_image::read_from_csv(test_filename);
        img_omp.print(", ", std::cout);
        
        std::cout << "\nTesting Execution Policy Read (std::execution::par):\n";
        auto img_par = TinyDIP::double_image::read_from_csv(std::execution::par, test_filename);
        img_par.print(", ", std::cout);

        // Clean up
        std::filesystem::remove(test_filename);
        return EXIT_SUCCESS;
    }
    else if(argc == 2)
    {
        std::filesystem::path source_filename = std::string(argv[1]);
        std::cout << "Read image: " << source_filename.string() << '\n';
        auto img_omp = TinyDIP::double_image::read_from_csv(source_filename.string().c_str());
        img_omp = TinyDIP::multiplies(TinyDIP::normalize(img_omp), 255.0);
        img_omp = TinyDIP::lanczos_resample(std::execution::par, img_omp, 1080, 1920);
        TinyDIP::bmp_write(source_filename.stem().string(), TinyDIP::constructRGB(
            TinyDIP::im2uint8(img_omp),
            TinyDIP::im2uint8(img_omp),
            TinyDIP::im2uint8(img_omp)
        ));
    }
    else if(argc == 3)
    {
        std::filesystem::path source_filename = std::string(argv[1]);
        std::filesystem::path destination_filename = std::string(argv[2]);
        std::cout << "Read image: " << source_filename.string() << '\n';
        auto img_omp = TinyDIP::double_image::read_from_csv(source_filename.string().c_str());
        img_omp = TinyDIP::multiplies(TinyDIP::normalize(img_omp), 255.0);
        img_omp = TinyDIP::lanczos_resample(std::execution::par, img_omp, 1080, 1920);
        TinyDIP::bmp_write(destination_filename.stem().string(), TinyDIP::constructRGB(
            TinyDIP::im2uint8(img_omp),
            TinyDIP::im2uint8(img_omp),
            TinyDIP::im2uint8(img_omp)
        ));
    }

    return EXIT_SUCCESS;
}