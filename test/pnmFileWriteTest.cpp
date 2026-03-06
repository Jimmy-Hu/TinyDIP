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

void pnmFileWriteTest(const char* file_path)
{
    auto image_input = TinyDIP::bmp_read(file_path, true);
    const std::filesystem::path output_filename = "write_to_pnm";
    TinyDIP::pnm::write(image_input, output_filename);
}

int main()
{
    TinyDIP::Timer timer1;
    const char* test_file = "../InputImages/1.bmp";
    pnmFileWriteTest(test_file);
    return EXIT_SUCCESS;
}