/* Developed by Jimmy Hu */

#ifndef BASE_H
#define BASE_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <utility>


using BYTE = unsigned char;

struct RGB
{
    unsigned char channels[3];
};

using GrayScale = BYTE;

struct HSV
{
    double channels[3];    //  Range: 0 <= H < 360, 0 <= S <= 1, 0 <= V <= 255
};

struct BMPIMAGE
{
    char FILENAME[MAX_PATH];
    
    unsigned int XSIZE;
    unsigned int YSIZE;
    unsigned char FILLINGBYTE;
    unsigned char *IMAGE_DATA;
};
#endif
