/* Developed by Jimmy Hu */

#ifndef BASE_H
#define BASE_H

#include <cmath>
#include <cstdbool>
#include <cstdio>
#include <cstdlib>
#include <string>

#define MAX_PATH 256
#define FILE_ROOT_PATH "./"

typedef unsigned char BYTE;

typedef struct RGB
{
    unsigned char channels[3];
};

typedef BYTE GrayScale;

typedef struct HSV
{
    long double channels[3];    //  Range: 0 <= H < 360, 0 <= S <= 1, 0 <= V <= 255
};

#endif
