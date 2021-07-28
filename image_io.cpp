/* Developed by Jimmy Hu */

#include "image_io.h"

namespace TinyDIP
{
    Image<RGB> raw_image_to_array(const int xsize, const int ysize, const unsigned char * const image)
    {
        auto output = Image<RGB>(xsize, ysize);  
        unsigned char FillingByte;
        FillingByte = bmp_filling_byte_calc(xsize);
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
            fname_bmp = filename;
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
        char fname_bmp[MAX_PATH];
        if(extension == false)
        {    
            sprintf(fname_bmp, "%s.bmp", filename);
        }        
        else
        {    
            strcpy(fname_bmp,filename);
        }    
        FILE *fp;
        fp = fopen(fname_bmp, "rb");
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

    //---- bmp_file_read function ---- 
    char bmp_read_detail(unsigned char * const image, const int xsize, const int ysize, const char * const filename, const bool extension)
    {
        char fname_bmp[MAX_PATH];
        if(extension == false)
        {    
            sprintf(fname_bmp, "%s.bmp", filename);
        }        
        else
        {    
            strcpy(fname_bmp,filename);
        }    
        unsigned char filling_bytes;
        filling_bytes = bmp_filling_byte_calc(xsize);
        FILE *fp;
        fp = fopen(fname_bmp, "rb");
        if (fp == NULL)
        {     
            printf("Fail to read file!\n");
            return -1;
        }             
        unsigned char header[54];
        fread(header, sizeof(unsigned char), 54, fp);
        fread(image, sizeof(unsigned char), (size_t)(long)(xsize * 3 + filling_bytes)*ysize, fp);
        fclose(fp); 
        return 0;
    }

    BMPIMAGE bmp_file_read(const char * const filename, const bool extension)
    {
        BMPIMAGE output;
        strcpy(output.FILENAME, "");
        output.XSIZE = 0;
        output.YSIZE = 0;
        output.IMAGE_DATA = NULL;
        if(filename == NULL)
        {    
            std::cerr << "Path is null\n";
            return output;
        }
        char fname_bmp[MAX_PATH];
        if(extension == false)
        {
            sprintf(fname_bmp, "%s.bmp", filename);
        }
        else
        {    
            strcpy(fname_bmp,filename);
        }    
        FILE *fp;
        fp = fopen(fname_bmp, "rb");
        if (fp == NULL)
        {     
            std::cerr << "Fail to read file!\n";
            return output;
        }             
        strcpy(output.FILENAME, fname_bmp);
        int OriginSizeX = bmp_read_x_size(output.FILENAME,true);
        int OriginSizeY = bmp_read_y_size(output.FILENAME,true);
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
            output.YSIZE = abs(OriginSizeY);
            printf("Width of the input image: %d\n", output.XSIZE);
            printf("Height of the input image: %d\n", output.YSIZE);
            printf("Size of the input image(Byte): %d\n", (size_t)output.XSIZE * output.YSIZE * 3);
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
            bmp_read_detail(OriginImageData, output.XSIZE, output.YSIZE, output.FILENAME, true);
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
        printf("Size of the input image(Byte): %d\n", (size_t)output.XSIZE * output.YSIZE * 3);
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
        bmp_read_detail(output.IMAGE_DATA, output.XSIZE, output.YSIZE, output.FILENAME, true);
        return output;
    }

    Image<RGB> bmp_read(const char* filename, const bool extension)
    {
        auto image = bmp_file_read(filename, extension);
        auto output = TinyDIP::raw_image_to_array(image.XSIZE, image.YSIZE, image.IMAGE_DATA);
        free(image.IMAGE_DATA);
        return output;
    }

    //----bmp_write function---- 
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
        char fname_bmp[MAX_PATH];
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
        sprintf(fname_bmp, "%s.bmp", filename);
        FILE *fp; 
        if (!(fp = fopen(fname_bmp, "wb")))
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

    unsigned char bmp_filling_byte_calc(const unsigned int xsize)
    {
        unsigned char filling_bytes;
        filling_bytes = (xsize % 4);
        return filling_bytes;
    }

}


