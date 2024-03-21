/* Developed by Jimmy Hu */

#include <string.h>
#include "image_io.h"

namespace TinyDIP
{
    Image<RGB> raw_image_to_array(const int xsize, const int ysize, const unsigned char * const image)
    {
        auto output = Image<RGB>(xsize, ysize);  
        unsigned char FillingByte = bmp_filling_byte_calc(xsize);
        for(int y = 0; y < ysize; y++)
        {
            for(int x = 0; x < xsize; x++)
            {
                for (int color = 0; color < 3; color++) {
                    output.at(x, y).channels[color] =
                        image[3 * (y * xsize + x) + y * FillingByte + (2 - color)];
                }
            }
        }
        return output;
    }

    //----bmp_read_x_size function definition----
    unsigned long bmp_read_x_size(const char *filename, const bool extension)
    {
        std::filesystem::path fname_bmp;
        if(extension == false)
        {    
            fname_bmp = std::string(filename) + ".bmp";
        }        
        else
        {    
            fname_bmp = std::string(filename);
        }    
        FILE *fp;
        fp = fopen(fname_bmp.string().c_str(), "rb");
        if (fp == NULL) 
        {     
            printf("Fail to read file!\n");
            return -1;
        }             
        unsigned char header[54];
        fread(header, sizeof(unsigned char), 54, fp);
        unsigned long output;
        output = header[18] + 
            ((unsigned long)header[19] << 8) +
            ((unsigned long)header[20] << 16) +
            ((unsigned long)header[21] << 24);
        fclose(fp);
        return output;
    }

    //---- bmp_read_y_size function ----
    unsigned long bmp_read_y_size(const char * const filename, const bool extension)
    {
        std::filesystem::path fname_bmp;
        if(extension == false)
        {    
            fname_bmp = std::string(filename) + ".bmp";
        }        
        else
        {    
            fname_bmp = std::string(filename);
        }    
        FILE *fp;
        fp = fopen(fname_bmp.string().c_str(), "rb");
        if (fp == NULL)
        {
            printf("Fail to read file!\n");
            return -1;
        }             
        unsigned char header[54];
        fread(header, sizeof(unsigned char), 54, fp);
        unsigned long output; 
        output= header[22] + 
            ((unsigned long)header[23] << 8) +
            ((unsigned long)header[24] << 16) +
            ((unsigned long)header[25] << 24);
        fclose(fp);
        return output;
    }

    //---- bmp_file_read function implementation ---- 
    char bmp_read_detail(unsigned char * const image, const int xsize, const int ysize, const char * const filename, const bool extension)
    {
        std::filesystem::path fname_bmp;
        if(extension == false)
        {    
            fname_bmp = std::string(filename) + ".bmp";
        }
        else
        {    
            fname_bmp = std::string(filename);
        }    
        unsigned char filling_bytes;
        filling_bytes = bmp_filling_byte_calc(xsize);
        FILE *fp;
        fp = fopen(fname_bmp.string().c_str(), "rb");
        if (fp == NULL)
        {     
            printf("Fail to read file!\n");
            return -1;
        }             
        unsigned char header[54];
        auto result = fread(header, sizeof(unsigned char), 54, fp);
        result = fread(image, sizeof(unsigned char), (size_t)(long)(xsize * 3 + filling_bytes)*ysize, fp);
        fclose(fp); 
        return 0;
    }

    //  bmp_file_read function implementation
    BMPIMAGE bmp_file_read(const char * const filename, const bool extension)
    {
        BMPIMAGE output;
        output.XSIZE = 0;
        output.YSIZE = 0;
        output.IMAGE_DATA = NULL;
        if(filename == NULL)
        {    
            std::cerr << "Path is null\n";
            return output;
        }
        std::filesystem::path fname_bmp;
        if(extension == false)
        {
            fname_bmp = std::string(filename) + ".bmp";
        }
        else
        {    
            fname_bmp = std::string(filename);
        }    
        FILE *fp;
        fp = fopen(fname_bmp.string().c_str(), "rb");
        if (fp == NULL)
        {     
            std::cerr << "Fail to read file: " << fname_bmp.string() << "!\n";
            return output;
        }             
        output.FILENAME = fname_bmp;
        int OriginSizeX = bmp_read_x_size(output.FILENAME.string().c_str(),true);
        int OriginSizeY = bmp_read_y_size(output.FILENAME.string().c_str(),true);
        if( (OriginSizeX == -1) || (OriginSizeY == -1) )
        {     
            std::cerr << "Fail to fetch information of image!";
            return output;
        }
        //  Deal with the negative height
        if (OriginSizeY < 0)
        {
            std::cout << "Deal with the negative height\n";
            output.XSIZE = OriginSizeX;
            output.YSIZE = std::abs(OriginSizeY);
            printf("Width of the input image: %d\n", output.XSIZE);
            printf("Height of the input image: %d\n", output.YSIZE);
            printf("Size of the input image(Byte): %ld\n", static_cast<size_t>(output.XSIZE * output.YSIZE * 3));
            output.FILLINGBYTE = bmp_filling_byte_calc(output.XSIZE);
            output.IMAGE_DATA = static_cast<unsigned char*>(malloc(sizeof *output.IMAGE_DATA * (output.XSIZE * 3 + output.FILLINGBYTE) * output.YSIZE));
            if (output.IMAGE_DATA == NULL)
            { 
                std::cerr << "Memory allocation error!";
                return output;
            }
            unsigned char *OriginImageData;
            OriginImageData = static_cast<unsigned char *>(malloc(sizeof *OriginImageData * (output.XSIZE * 3 + output.FILLINGBYTE) * output.YSIZE));
            if (OriginImageData == NULL)
            { 
                std::cerr << "Memory allocation error!";
                return output;
            }
            for(int i = 0; i < ((output.XSIZE * 3 + output.FILLINGBYTE) * output.YSIZE);i++)
            {
                OriginImageData[i] = 255;
            }
            bmp_read_detail(OriginImageData, output.XSIZE, output.YSIZE, output.FILENAME.string().c_str(), true);
            for (size_t loop = 0; loop < (output.XSIZE * 3 + output.FILLINGBYTE) * output.YSIZE; loop++)
            {
                printf("%d\n", OriginImageData[loop]);
                system("read -n 1 -s -p \"Press any key to continue...\n\"");
            }
            for(int y = 0; y < output.YSIZE; y++)
            {
                for(int x = 0; x < output.XSIZE; x++)
                {
                    for (int color = 0; color < 3; color++) {
                        output.IMAGE_DATA[3 * (y * output.XSIZE + x) + y * output.FILLINGBYTE + (2 - color)] = 
                            OriginImageData[3 * ((output.YSIZE - y - 1) * output.XSIZE + x) + (output.YSIZE - y - 1) * output.FILLINGBYTE + (2 - color)];
                    }
                    printf("Pixel (%d, %d): %d, %d, %d\n", x, y, 
                    OriginImageData[3 * ((output.YSIZE - y - 1) * output.XSIZE + x) + 2],
                    OriginImageData[3 * ((output.YSIZE - y - 1) * output.XSIZE + x) + 1],
                    OriginImageData[3 * ((output.YSIZE - y - 1) * output.XSIZE + x) + 0]);
                    system("read -n 1 -s -p \"Press any key to continue...\n\"");
                }
            }
            free(OriginImageData);
            return output;
        }

        output.XSIZE = (unsigned int)OriginSizeX;
        output.YSIZE = (unsigned int)OriginSizeY;
        printf("Width of the input image: %d\n",output.XSIZE);
        printf("Height of the input image: %d\n",output.YSIZE);
        printf("Size of the input image(Byte): %ld\n", static_cast<size_t>(output.XSIZE * output.YSIZE * 3));
        output.FILLINGBYTE = bmp_filling_byte_calc(output.XSIZE);
        output.IMAGE_DATA = static_cast<unsigned char *>(malloc(sizeof *output.IMAGE_DATA * (output.XSIZE * 3 + output.FILLINGBYTE) * output.YSIZE));
        if (output.IMAGE_DATA == NULL)
        { 
            std::cerr << "Memory allocation error!";
            return output;
        }     
        for(int i = 0; i < ((output.XSIZE * 3 + output.FILLINGBYTE) * output.YSIZE);i++)
        {
            output.IMAGE_DATA[i] = 255;
        }
        bmp_read_detail(output.IMAGE_DATA, output.XSIZE, output.YSIZE, output.FILENAME.string().c_str(), true);
        return output;
    }

    Image<RGB> bmp_read(const char* filename, const bool extension = true)
    {
        auto image = bmp_file_read(filename, extension);
        auto output = TinyDIP::raw_image_to_array(image.XSIZE, image.YSIZE, image.IMAGE_DATA);
        free(image.IMAGE_DATA);
        return output;
    }

    Image<RGB> bmp_read(std::string filename, const bool extension = true)
    {
        return bmp_read(filename.c_str(), extension);
    }

    //----bmp_write function---- 
    int bmp_write(std::string filename, Image<RGB> input)
    {
        return bmp_write(filename.c_str(), input);
    }

    int bmp_write(const char *filename, Image<RGB> input)
    {
        auto image_data = TinyDIP::array_to_raw_image(input);
        auto sizex = input.getWidth();
        auto sizey = input.getHeight();
        auto result = TinyDIP::bmp_write(filename, sizex, sizey, image_data);
        free(image_data);
        return result;
    }
    
    int bmp_write(const char * const filename, const int xsize, const int ysize, const unsigned char * const image) 
    {
        unsigned char FillingByte;
        FillingByte = bmp_filling_byte_calc(xsize);
        unsigned char header[54] =
        {
        0x42, 0x4d, 0, 0, 0, 0, 0, 0, 0, 0,
        54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0                        
        };                                
        unsigned long file_size = (long)xsize * (long)ysize * 3 + 54;
        unsigned long width, height;
        std::filesystem::path fname_bmp;
        header[2] = (unsigned char)(file_size &0x000000ff);
        header[3] = (file_size >> 8) & 0x000000ff;        
        header[4] = (file_size >> 16) & 0x000000ff;        
        header[5] = (file_size >> 24) & 0x000000ff;        
        
        width = xsize;
        header[18] = width & 0x000000ff;
        header[19] = (width >> 8) &0x000000ff;
        header[20] = (width >> 16) &0x000000ff;
        header[21] = (width >> 24) &0x000000ff;
        
        height = ysize;
        header[22] = height &0x000000ff;
        header[23] = (height >> 8) &0x000000ff;
        header[24] = (height >> 16) &0x000000ff;
        header[25] = (height >> 24) &0x000000ff;
        fname_bmp = std::string(filename) + ".bmp";
        FILE *fp; 
        if (!(fp = fopen(fname_bmp.string().c_str(), "wb")))
        {    
            return -1;
        }        
        fwrite(header, sizeof(unsigned char), 54, fp);
        fwrite(image, sizeof(unsigned char), (size_t)(long)(xsize * 3 + FillingByte) * ysize, fp);
        fclose(fp);
        return 0;
    }

    unsigned char *array_to_raw_image(Image<RGB> input)
    {
        std::size_t xsize = input.getWidth();
        std::size_t ysize = input.getHeight();
        unsigned char FillingByte;
        FillingByte = bmp_filling_byte_calc(xsize);
        unsigned char *output;
        output = static_cast<unsigned char *>(malloc(sizeof *output * (xsize * 3 + FillingByte) * ysize));
        if(output == NULL)
        {    
            std::cerr << "Memory allocation error!";
            return NULL;
        }
        for(int y = 0;y < ysize;y++)
        {
            for(int x = 0;x < (xsize * 3 + FillingByte);x++)
            {
                output[y * (xsize * 3 + FillingByte) + x] = 0;
            }
        }
        for(int y = 0;y<ysize;y++)
        {
            for(int x = 0;x<xsize;x++)
            {
                for (int color = 0; color < 3; color++) {
                    output[3 * (y * xsize + x) + y * FillingByte + (2 - color)]
                    = input.at(x, y).channels[color];
                }
            }
        }
        return output;
    }

    unsigned char bmp_filling_byte_calc(const unsigned int xsize, const int mod_num)
    {
        unsigned char filling_bytes;
        filling_bytes = (xsize % mod_num);
        return filling_bytes;
    }

    namespace double_image
    {
        double* array_to_raw_image(Image<double> input)
        {
            std::size_t xsize = input.getWidth();
            std::size_t ysize = input.getHeight();
            unsigned char FillingByte;
            FillingByte = bmp_filling_byte_calc(xsize, 8);
            double* output;
            output = static_cast<double*>(malloc(sizeof * output * (xsize + FillingByte) * ysize));
            if (output == NULL)
            {
                std::cerr << "Memory allocation error!";
                return NULL;
            }
            for (int y = 0; y < ysize; y++)
            {
                for (int x = 0; x < xsize; x++)
                {
                    output[(y * xsize + x) + y * FillingByte]
                        = input.at(x, y);
                }
            }
            return output;
        }
        
        int write(const char* filename, const int xsize, const int ysize, const double* const image)
        {
            unsigned char FillingByte;
            FillingByte = bmp_filling_byte_calc(xsize, 8);
            unsigned char header[54] =
            {
            0x42, 0x4d, 0, 0, 0, 0, 0, 0, 0, 0,
            54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0
            };
            unsigned long file_size = (long)xsize * (long)ysize + 54;
            unsigned long width, height;
            std::filesystem::path fname_bmp;
            header[2] = (unsigned char)(file_size & 0x000000ff);
            header[3] = (file_size >> 8) & 0x000000ff;
            header[4] = (file_size >> 16) & 0x000000ff;
            header[5] = (file_size >> 24) & 0x000000ff;

            width = xsize;
            header[18] = width & 0x000000ff;
            header[19] = (width >> 8) & 0x000000ff;
            header[20] = (width >> 16) & 0x000000ff;
            header[21] = (width >> 24) & 0x000000ff;

            height = ysize;
            header[22] = height & 0x000000ff;
            header[23] = (height >> 8) & 0x000000ff;
            header[24] = (height >> 16) & 0x000000ff;
            header[25] = (height >> 24) & 0x000000ff;
            fname_bmp = std::string(filename) + ".dbmp";
            FILE* fp;
            if (!(fp = fopen(fname_bmp.string().c_str(), "wb")))
            {
                return -1;
            }
            fwrite(header, sizeof(unsigned char), 54, fp);
            fwrite(image, sizeof(double), (size_t)(long)(xsize + FillingByte) * ysize, fp);
            fclose(fp);
            return 0;
        }

        int write(const char* filename, Image<double> input)
        {
            auto image_data = TinyDIP::double_image::array_to_raw_image(input);
            auto sizex = input.getWidth();
            auto sizey = input.getHeight();
            auto result = TinyDIP::double_image::write(filename, sizex, sizey, image_data);
            free(image_data);
            return result;
        }

        //  read function implementation
        TinyDIP::Image<double> read(const char* const filename, const bool extension)
        {
            std::filesystem::path target_path;
            if (extension == false)
            {
                target_path = std::string(filename) + ".dbmp";
            }
            else
            {
                target_path = std::string(filename);
            }
            FILE* fp;
            fp = fopen(target_path.string().c_str(), "rb");
            if (fp == NULL)
            {
                std::cerr << "Fail to read file: " << target_path.string() << "!\n";
                return TinyDIP::Image<double>(0, 0);
            }
            int OriginSizeX = bmp_read_x_size(target_path.string().c_str(), true);
            int OriginSizeY = bmp_read_y_size(target_path.string().c_str(), true);
            if ((OriginSizeX == -1) || (OriginSizeY == -1))
            {
                std::cerr << "Fail to fetch information of image!";
                return TinyDIP::Image<double>(0, 0);
            }
            printf("Width of the input image: %d\n", OriginSizeX);
            printf("Height of the input image: %d\n", OriginSizeY);
            printf("Size of the input image: %ld\n", static_cast<size_t>(OriginSizeX * OriginSizeY));
            unsigned char header[54];
            auto returnValue = fread(header, sizeof(unsigned char), 54, fp);
            double* image;
            unsigned char FillingByte = bmp_filling_byte_calc(OriginSizeX, 8);
            image = static_cast<double*>(malloc(sizeof * image * (OriginSizeX + FillingByte) * OriginSizeY));
            auto returnValue = fread(image, sizeof(double), (size_t)(double)(OriginSizeX + FillingByte) * OriginSizeY, fp);
            TinyDIP::Image<double> output(OriginSizeX, OriginSizeY);
            for (int y = 0; y < OriginSizeY; y++)
            {
                for (int x = 0; x < OriginSizeX; x++)
                {
                    output.at(x, y) =
                        image[(y * OriginSizeX + x) + y * FillingByte];
                }
            }
            free(image);
            fclose(fp);
            return output;
        }

        double* array_to_raw_image(Image<HSV> input)
        {
            std::size_t xsize = input.getWidth();
            std::size_t ysize = input.getHeight();
            unsigned char FillingByte;
            FillingByte = bmp_filling_byte_calc(xsize, 8);
            double* output;
            output = static_cast<double*>(malloc(sizeof * output * (xsize * 3 + FillingByte) * ysize));
            if (output == NULL)
            {
                std::cerr << "Memory allocation error!";
                return NULL;
            }
            for (int y = 0; y < ysize; y++)
            {
                for (int x = 0; x < xsize; x++)
                {
                    for (int channel_index = 0; channel_index < 3; channel_index++) {
                        output[3 * (y * xsize + x) + y * FillingByte + (2 - channel_index)]
                            = input.at(x, y).channels[channel_index];
                    }
                }
            }
            return output;
        }
    }

    int hsv_write_detail(const char* const filename, const int xsize, const int ysize, const double* const image)
    {
        unsigned char FillingByte = bmp_filling_byte_calc(xsize, 8);
        unsigned char header[54] =
        {
        0x42, 0x4d, 0, 0, 0, 0, 0, 0, 0, 0,
        54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0
        };
        unsigned long file_size = (long)xsize * (long)ysize * 3 + 54;
        unsigned long width, height;
        std::filesystem::path fname_bmp;
        header[2] = (unsigned char)(file_size & 0x000000ff);
        header[3] = (file_size >> 8) & 0x000000ff;
        header[4] = (file_size >> 16) & 0x000000ff;
        header[5] = (file_size >> 24) & 0x000000ff;

        width = xsize;
        header[18] = width & 0x000000ff;
        header[19] = (width >> 8) & 0x000000ff;
        header[20] = (width >> 16) & 0x000000ff;
        header[21] = (width >> 24) & 0x000000ff;

        height = ysize;
        header[22] = height & 0x000000ff;
        header[23] = (height >> 8) & 0x000000ff;
        header[24] = (height >> 16) & 0x000000ff;
        header[25] = (height >> 24) & 0x000000ff;
        fname_bmp = std::string(filename) + ".hsv";
        FILE* fp;
        if (!(fp = fopen(fname_bmp.string().c_str(), "wb")))
        {
            return -1;
        }
        fwrite(header, sizeof(unsigned char), 54, fp);
        fwrite(image, sizeof(double), (size_t)(long)(xsize * 3 + FillingByte) * ysize, fp);
        fclose(fp);
        return 0;
    }

    int hsv_write(const char* const filename, Image<HSV> input)
    {
        auto image_data = TinyDIP::double_image::array_to_raw_image(input);
        auto sizex = input.getWidth();
        auto sizey = input.getHeight();
        auto result = TinyDIP::hsv_write_detail(filename, sizex, sizey, image_data);
        free(image_data);
        return result;
    }

    //  hsv_read function implementation
    Image<HSV> hsv_read(const char* const filename, const bool extension)
    {
        std::filesystem::path target_path;
        if (extension == false)
        {
            target_path = std::string(filename) + ".hsv";
        }
        else
        {
            target_path = std::string(filename);
        }
        FILE* fp;
        fp = fopen(target_path.string().c_str(), "rb");
        if (fp == NULL)
        {
            std::cerr << "Fail to read file: " << target_path.string() << "!\n";
            return Image<HSV>(0, 0);
        }
        int OriginSizeX = bmp_read_x_size(target_path.string().c_str(), true);
        int OriginSizeY = bmp_read_y_size(target_path.string().c_str(), true);
        if ((OriginSizeX == -1) || (OriginSizeY == -1))
        {
            std::cerr << "Fail to fetch information of image!";
            return Image<HSV>(0, 0);
        }
        printf("Width of the input image: %d\n", OriginSizeX);
        printf("Height of the input image: %d\n", OriginSizeY);
        printf("Size of the input image: %ld\n", static_cast<size_t>(OriginSizeX * OriginSizeY * 3));
        unsigned char header[54];
        auto returnValue = fread(header, sizeof(unsigned char), 54, fp);
        double* image;
        unsigned char filling_bytes = bmp_filling_byte_calc(OriginSizeX, 8);
        image = static_cast<double*>(malloc(sizeof * image * (OriginSizeX * 3 + filling_bytes) * OriginSizeY));
        auto returnValue = fread(image, sizeof(double), (size_t)(long)(OriginSizeX * 3 + filling_bytes) * OriginSizeY, fp);
        fclose(fp);
        auto output = Image<HSV>(OriginSizeX, OriginSizeY);
        for (int y = 0; y < OriginSizeY; y++)
        {
            for (int x = 0; x < OriginSizeX; x++)
            {
                for (int channel_index = 0; channel_index < 3; channel_index++) {
                    output.at(x, y).channels[channel_index] =
                        image[3 * (y * OriginSizeX + x) + y * filling_bytes + (2 - channel_index)];
                }
            }
        }
        free(image);
        return output;
    }
}