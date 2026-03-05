//  Developed by Jimmy Hu

#include <cassert>
#include <execution>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../image_io.h"
#include "../cube.h"
#include "../cube_operations.h"
#include "../timer.h"

void pnmFileReadTest(const std::filesystem::path& file_path)
{
    auto image_input = TinyDIP::pnm::read(file_path);
    TinyDIP::bmp_write("read_from_pnm", image_input);
}

int main()
{
    TinyDIP::Timer timer1;
    const std::filesystem::path test_file = "ImgTest_0015.ppm";
    pnmFileReadTest(test_file);
    return EXIT_SUCCESS;
}