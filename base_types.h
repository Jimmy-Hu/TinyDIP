/* Developed by Jimmy Hu */

#ifndef BASE_H
#define BASE_H

#include <cstdint>
#include <filesystem>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <utility>

using BYTE = unsigned char;

struct RGB
{
    BYTE channels[3];
};

using GrayScale = BYTE;

struct HSV
{
    double channels[3];    //  Range: 0 <= H < 360, 0 <= S <= 1, 0 <= V <= 255
};

struct BMPIMAGE
{
    std::filesystem::path FILENAME;
    
    unsigned int XSIZE;
    unsigned int YSIZE;
    BYTE FILLINGBYTE;
    BYTE *IMAGE_DATA;
};
#endif
