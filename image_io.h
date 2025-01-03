/* Developed by Jimmy Hu */

#ifndef TINYDIP_IMAGE_IO_H       // image_io.h header guard, follow the suggestion from https://codereview.stackexchange.com/a/293832/231235
#define TINYDIP_IMAGE_IO_H

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include "image.h"

namespace TinyDIP
{
    Image<RGB> raw_image_to_array(const int xsize, const int ysize, const unsigned char * const image);

    unsigned long bmp_read_x_size(const char *filename, const bool extension);

    unsigned long bmp_read_y_size(const char *filename, const bool extension);

    char bmp_read_detail(unsigned char *image, const int xsize, const int ysize, const char *filename, const bool extension);

    BMPIMAGE bmp_file_read(const char *filename, const bool extension);

    Image<RGB> bmp_read(const char* filename, const bool extension);

    int bmp_write(std::string filename, Image<RGB> input);

    int bmp_write(const char *filename, Image<RGB> input);

    int bmp_write(const char *filename, const int xsize, const int ysize, const unsigned char *image);

    unsigned char *array_to_raw_image(Image<RGB> input);

    unsigned char bmp_filling_byte_calc(const unsigned int xsize, const int mod_num = 4);

    namespace double_image
    {
        double* array_to_raw_image(Image<double> input);

        int write(const char* filename, const int xsize, const int ysize, const double* image);

        int write(const char* filename, Image<double> input);

        TinyDIP::Image<double> read(const char* const filename, const bool extension);

        double* array_to_raw_image(Image<HSV> input);
    }

    int hsv_write_detail(const char* const filename, const int xsize, const int ysize, const double* const image);

    int hsv_write(const char* const filename, Image<HSV> input);

    Image<HSV> hsv_read(const char* const filename, const bool extension);
}

#endif

