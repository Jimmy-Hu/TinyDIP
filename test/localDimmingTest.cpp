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

//  RGB_max Function Implementation
static auto RGB_max(const TinyDIP::Image<TinyDIP::RGB>& input_image)
{
    return TinyDIP::pixelwise_transform([](auto&& each_pixel)
            {
                auto max_value = std::ranges::max(each_pixel.channels);
                TinyDIP::RGB new_pixel{ max_value, max_value, max_value };
                each_pixel = new_pixel;
            }, input_image)
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    std::cout << "argc = " << std::to_string(argc) << '\n';
    if(argc == 2)
    {
        auto input_path = std::string(argv[1]);
        auto input_img = TinyDIP::bmp_read(input_path.c_str(), true);
        auto RGB_max_result = RGB_max(input_img);
    }
}



