/* Developed by Jimmy Hu */

#include <chrono>
#include <execution>
#include <map>
#include <omp.h>
#include <span>
#include <sstream>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    std::cout << "argc parameter: " << std::to_string(argc) << '\n';

    std::string file_path = "../InputImages/boat1";
    auto bmp1 = TinyDIP::bmp_read(file_path.c_str(), false);
    file_path = "../InputImages/boat2";
    auto bmp2 = TinyDIP::bmp_read(file_path.c_str(), false);
    if ( (bmp1.getWidth() == 0) || (bmp1.getHeight() == 0) || (bmp2.getWidth() == 0) || (bmp2.getHeight() == 0) )
    {
        std::cerr << "Fail to read image\n";
        return EXIT_FAILURE;
    }
    std::cout << "Image 1 size: " << bmp1.getWidth() << " x " << bmp1.getHeight() << "\n";
    std::cout << "Image 2 size: " << bmp2.getWidth() << " x " << bmp2.getHeight() << "\n";
    auto warped_img2 = TinyDIP::imstitch(bmp1, bmp2);
    //file_path = "../OutputImages/SIFT_Stitching_Result.bmp";
    file_path = "../OutputImages/SIFT_Stitching_Result_Boat";
    if (!TinyDIP::bmp_write(file_path.c_str(), warped_img2))
    {
        std::cerr << "Fail to write image\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}