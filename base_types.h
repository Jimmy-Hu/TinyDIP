/* Developed by Jimmy Hu */

#ifndef TINYDIP_BASE_TYPES_H  // base_types.h header guard, follow the suggestion from https://codereview.stackexchange.com/a/293832/231235
#define TINYDIP_BASE_TYPES_H

#include <cstdint>
#include <filesystem>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <utility>

namespace TinyDIP
{
    struct RGB
    {
        std::uint8_t channels[3];
    };

    struct RGB_DOUBLE
    {
        double channels[3];
    };

    using GrayScale = std::uint8_t;

    struct HSV
    {
        double channels[3];    //  Range: 0 <= H < 360, 0 <= S <= 1, 0 <= V <= 255
    };

struct BMPIMAGE
{
    std::filesystem::path FILENAME;
    
    unsigned int XSIZE;
    unsigned int YSIZE;
    std::uint8_t FILLINGBYTE;
    std::uint8_t*IMAGE_DATA;
};
#endif
