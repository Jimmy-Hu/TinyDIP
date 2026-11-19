/* Developed by Jimmy Hu */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "image_io.h"

namespace TinyDIP
{
    namespace
    {
        std::filesystem::path resolve_path_with_ext(
            const char* const filename,
            const bool has_extension,
            const char* const default_ext
        )
        {
            return has_extension
                ? std::filesystem::path(std::string(filename))
                : std::filesystem::path(std::string(filename) + default_ext);
        }

        constexpr unsigned long read_le_u32(const unsigned char* const bytes) noexcept
        {
            return static_cast<unsigned long>(bytes[0]) +
                   (static_cast<unsigned long>(bytes[1]) << 8) +
                   (static_cast<unsigned long>(bytes[2]) << 16) +
                   (static_cast<unsigned long>(bytes[3]) << 24);
        }

        constexpr void write_le_u32(unsigned char* const dst, const unsigned long val) noexcept
        {
            dst[0] = static_cast<unsigned char>(val & 0x000000ffUL);
            dst[1] = static_cast<unsigned char>((val >> 8) & 0x000000ffUL);
            dst[2] = static_cast<unsigned char>((val >> 16) & 0x000000ffUL);
            dst[3] = static_cast<unsigned char>((val >> 24) & 0x000000ffUL);
        }

        unsigned long bmp_read_header_u32(
            const char* const filename,
            const bool extension,
            const std::size_t byte_offset
        )
        {
            const auto fname_bmp = resolve_path_with_ext(filename, extension, ".bmp");
            FILE* const fp = std::fopen(fname_bmp.string().c_str(), "rb");
            if (fp == nullptr)
            {
                std::printf("Fail to read file!\n");
                return static_cast<unsigned long>(-1);
            }
            unsigned char header[54]{};
            const auto bytes_read = std::fread(header, sizeof(unsigned char), 54, fp);
            std::fclose(fp);
            if (bytes_read != 54)
            {
                return static_cast<unsigned long>(-1);
            }
            return read_le_u32(header + byte_offset);
        }

        template <typename ElemT>
        int write_bmp_container_file(
            const char* const filename,
            const char* const ext,
            const int xsize,
            const int ysize,
            const std::size_t channels_per_pixel,
            const int pad_mod,
            const ElemT* const raw_data
        )
        {
            const unsigned char filling = bmp_filling_byte_calc(static_cast<unsigned int>(xsize), pad_mod);
            unsigned char header[54] = {
                0x42, 0x4d, 0, 0, 0, 0, 0, 0, 0, 0,
                54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0
            };
            const unsigned long file_size =
                static_cast<unsigned long>(xsize) * static_cast<unsigned long>(ysize) *
                static_cast<unsigned long>(channels_per_pixel) + 54UL;
            write_le_u32(header + 2, file_size);
            write_le_u32(header + 18, static_cast<unsigned long>(xsize));
            write_le_u32(header + 22, static_cast<unsigned long>(ysize));

            const std::filesystem::path out_path = std::string(filename) + ext;
            FILE* const fp = std::fopen(out_path.string().c_str(), "wb");
            if (fp == nullptr)
            {
                return -1;
            }
            std::fwrite(header, sizeof(unsigned char), 54, fp);
            const std::size_t row_elems = static_cast<std::size_t>(xsize) * channels_per_pixel + filling;
            std::fwrite(raw_data, sizeof(ElemT), row_elems * static_cast<std::size_t>(ysize), fp);
            std::fclose(fp);
            return 0;
        }

        template <typename RawElemT, typename PixelT, typename WritePixelFn>
        RawElemT* pack_padded_row_buffer(
            const Image<PixelT>& input,
            const std::size_t channels_per_pixel,
            const int pad_mod,
            const WritePixelFn write_pixel
        )
        {
            const std::size_t xsize = input.getWidth();
            const std::size_t ysize = input.getHeight();
            const unsigned char filling = bmp_filling_byte_calc(static_cast<unsigned int>(xsize), pad_mod);
            const std::size_t row_stride = xsize * channels_per_pixel + filling;
            auto* const output = static_cast<RawElemT*>(std::calloc(row_stride * ysize, sizeof(RawElemT)));
            if (output == nullptr)
            {
                std::cerr << "Memory allocation error!";
                return nullptr;
            }
            for (std::size_t y = 0; y < ysize; ++y)
            {
                for (std::size_t x = 0; x < xsize; ++x)
                {
                    write_pixel(output + y * row_stride + x * channels_per_pixel, input.at_without_boundary_check(x, y));
                }
            }
            return output;
        }

        template <typename PixelT, typename RawElemT, typename ReadPixelFn>
        Image<PixelT> unpack_padded_row_buffer(
            const int xsize,
            const int ysize,
            const RawElemT* const raw_data,
            const std::size_t channels_per_pixel,
            const int pad_mod,
            const ReadPixelFn read_pixel
        )
        {
            const auto ux = static_cast<std::size_t>(xsize);
            const auto uy = static_cast<std::size_t>(ysize);
            Image<PixelT> output(ux, uy);
            const unsigned char filling = bmp_filling_byte_calc(static_cast<unsigned int>(xsize), pad_mod);
            const std::size_t row_stride = ux * channels_per_pixel + filling;
            for (std::size_t y = 0; y < uy; ++y)
            {
                for (std::size_t x = 0; x < ux; ++x)
                {
                    read_pixel(output.at_without_boundary_check(x, y), raw_data + y * row_stride + x * channels_per_pixel);
                }
            }
            return output;
        }

        template <typename PixelT, typename RawElemT>
        RawElemT* pack_3channel_bgr_buffer(const Image<PixelT>& input, const int pad_mod)
        {
            return pack_padded_row_buffer<RawElemT>(
                input,
                3,
                pad_mod,
                [](RawElemT* const dst, const PixelT& px) noexcept
                {
                    for (std::size_t c = 0; c < 3; ++c)
                    {
                        dst[2 - c] = static_cast<RawElemT>(px.channels[c]);
                    }
                }
            );
        }

        template <typename PixelT, typename RawElemT>
        Image<PixelT> unpack_3channel_bgr_buffer(
            const int xsize,
            const int ysize,
            const RawElemT* const raw_data,
            const int pad_mod
        )
        {
            return unpack_padded_row_buffer<PixelT>(
                xsize,
                ysize,
                raw_data,
                3,
                pad_mod,
                [](PixelT& px, const RawElemT* const src) noexcept
                {
                    using ChanT = std::remove_cvref_t<decltype(px.channels[0])>;
                    for (std::size_t c = 0; c < 3; ++c)
                    {
                        px.channels[c] = static_cast<ChanT>(src[2 - c]);
                    }
                }
            );
        }

        template <typename PixelT, typename PackFn, typename WriteFn>
        int write_packed_image(
            const char* const filename,
            Image<PixelT>&& input,
            const PackFn pack_fn,
            const WriteFn write_fn
        )
        {
            const auto sizex = static_cast<int>(input.getWidth());
            const auto sizey = static_cast<int>(input.getHeight());
            auto* const image_data = pack_fn(std::move(input));
            const auto result = write_fn(filename, sizex, sizey, image_data);
            std::free(image_data);
            return result;
        }

        bool read_double_container_payload(
            const char* const filename,
            const bool extension,
            const char* const default_ext,
            const std::size_t channels_per_pixel,
            int& out_x,
            int& out_y,
            double*& out_buf
        )
        {
            out_x = 0;
            out_y = 0;
            out_buf = nullptr;

            const auto target_path = resolve_path_with_ext(filename, extension, default_ext);
            FILE* const fp = std::fopen(target_path.string().c_str(), "rb");
            if (fp == nullptr)
            {
                std::cerr << "Fail to read file: " << target_path.string() << "!\n";
                return false;
            }
            const int origin_x = static_cast<int>(bmp_read_x_size(target_path.string().c_str(), true));
            const int origin_y = static_cast<int>(bmp_read_y_size(target_path.string().c_str(), true));
            if (origin_x == -1 || origin_y == -1)
            {
                std::fclose(fp);
                std::cerr << "Fail to fetch information of image!";
                return false;
            }
            std::printf("Width of the input image: %d\n", origin_x);
            std::printf("Height of the input image: %d\n", origin_y);
            std::printf(
                "Size of the input image: %zu\n",
                static_cast<std::size_t>(origin_x) * static_cast<std::size_t>(origin_y) * channels_per_pixel
            );

            unsigned char header[54]{};
            auto return_value = std::fread(header, sizeof(unsigned char), 54, fp);
            const unsigned char filling = bmp_filling_byte_calc(static_cast<unsigned int>(origin_x), 8);
            const std::size_t row_stride = static_cast<std::size_t>(origin_x) * channels_per_pixel + filling;
            const auto uy = static_cast<std::size_t>(origin_y);
            out_buf = static_cast<double*>(std::malloc(sizeof(double) * row_stride * uy));
            if (out_buf != nullptr)
            {
                return_value = std::fread(out_buf, sizeof(double), row_stride * uy, fp);
            }
            (void)return_value;
            std::fclose(fp);
            out_x = origin_x;
            out_y = origin_y;
            return out_buf != nullptr;
        }

        template <typename PixelT, typename UnpackFn>
        Image<PixelT> read_double_container_image(
            const char* const filename,
            const bool extension,
            const char* const default_ext,
            const std::size_t channels_per_pixel,
            const UnpackFn unpack_fn
        )
        {
            int origin_x = 0;
            int origin_y = 0;
            double* raw_buf = nullptr;
            if (!read_double_container_payload(filename, extension, default_ext, channels_per_pixel, origin_x, origin_y, raw_buf))
            {
                return Image<PixelT>(0, 0);
            }
            auto output = unpack_fn(origin_x, origin_y, raw_buf);
            std::free(raw_buf);
            return output;
        }
    }

    Image<RGB> raw_image_to_array(const int xsize, const int ysize, const unsigned char* const image)
    {
        return unpack_3channel_bgr_buffer<RGB>(xsize, ysize, image, 4);
    }

    unsigned long bmp_read_x_size(const char* const filename, const bool extension)
    {
        return bmp_read_header_u32(filename, extension, 18);
    }

    unsigned long bmp_read_y_size(const char* const filename, const bool extension)
    {
        return bmp_read_header_u32(filename, extension, 22);
    }

    char bmp_read_detail(unsigned char* const image, const int xsize, const int ysize, const char* const filename, const bool extension)
    {
        const auto fname_bmp = resolve_path_with_ext(filename, extension, ".bmp");
        const unsigned char filling_bytes = bmp_filling_byte_calc(static_cast<unsigned int>(xsize));
        FILE* const fp = std::fopen(fname_bmp.string().c_str(), "rb");
        if (fp == nullptr)
        {
            std::printf("Fail to read file!\n");
            return -1;
        }

        unsigned char header[54]{};
        auto result = std::fread(header, sizeof(unsigned char), 54, fp);
        if (result != 54)
        {
            std::fclose(fp);
            return -1;
        }

        const int biBitCount = header[28] + (header[29] << 8);
        const auto bfOffBits = static_cast<long>(read_le_u32(header + 10));

        if (biBitCount == 24)
        {
            std::fseek(fp, bfOffBits, SEEK_SET);
            const std::size_t total_bytes =
                (static_cast<std::size_t>(xsize) * 3 + filling_bytes) * static_cast<std::size_t>(ysize);
            result = std::fread(image, sizeof(unsigned char), total_bytes, fp);
            (void)result;
        }
        else if (biBitCount == 8)
        {
            unsigned char palette[1024] = { 0 };
            const unsigned int header_size = static_cast<unsigned int>(read_le_u32(header + 14));
            int palette_size = static_cast<int>(bfOffBits) - static_cast<int>(14 + header_size);
            if (palette_size > 1024) palette_size = 1024;

            if (palette_size > 0)
            {
                std::fseek(fp, static_cast<long>(14 + header_size), SEEK_SET);
                result = std::fread(palette, 1, static_cast<std::size_t>(palette_size), fp);
            }

            std::fseek(fp, bfOffBits, SEEK_SET);
            const int padding_8 = (4 - (xsize % 4)) % 4;
            const auto row_in_bytes = static_cast<std::size_t>(xsize + padding_8);
            const std::size_t out_row_stride = static_cast<std::size_t>(xsize) * 3 + filling_bytes;

            auto* const row_buffer = static_cast<unsigned char*>(std::malloc(row_in_bytes));
            if (row_buffer != nullptr)
            {
                for (int y = 0; y < ysize; ++y)
                {
                    std::memset(row_buffer, 0, row_in_bytes);
                    result = std::fread(row_buffer, 1, row_in_bytes, fp);
                    const auto uy = static_cast<std::size_t>(y);
                    for (int x = 0; x < xsize; ++x)
                    {
                        const unsigned char index = row_buffer[x];
                        const auto ux = static_cast<std::size_t>(x);
                        image[uy * out_row_stride + ux * 3 + 0] = palette[index * 4 + 0];
                        image[uy * out_row_stride + ux * 3 + 1] = palette[index * 4 + 1];
                        image[uy * out_row_stride + ux * 3 + 2] = palette[index * 4 + 2];
                    }
                    for (int p = 0; p < filling_bytes; ++p)
                    {
                        image[uy * out_row_stride + static_cast<std::size_t>(xsize) * 3 + static_cast<std::size_t>(p)] = 0;
                    }
                }
                std::free(row_buffer);
            }
        }
        else
        {
            std::printf("Unsupported BMP bit depth: %d\n", biBitCount);
        }

        std::fclose(fp);
        return 0;
    }

    BMPIMAGE bmp_file_read(const char* const filename, const bool extension)
    {
        BMPIMAGE output{};
        output.XSIZE = 0;
        output.YSIZE = 0;
        output.IMAGE_DATA = nullptr;
        if (filename == nullptr)
        {
            std::cerr << "Path is null\n";
            return output;
        }
        const auto fname_bmp = resolve_path_with_ext(filename, extension, ".bmp");
        FILE* const fp = std::fopen(fname_bmp.string().c_str(), "rb");
        if (fp == nullptr)
        {
            std::cerr << "Fail to read file: " << fname_bmp.string() << "!\n";
            throw std::runtime_error("Fail to read BMP file!");
        }
        std::fclose(fp);

        output.FILENAME = fname_bmp;
        const int origin_x = static_cast<int>(bmp_read_x_size(output.FILENAME.string().c_str(), true));
        const int origin_y = static_cast<int>(bmp_read_y_size(output.FILENAME.string().c_str(), true));
        if (origin_x == -1 || origin_y == -1)
        {
            std::cerr << "Fail to fetch information of image!";
            return output;
        }

        const bool top_down = (origin_y < 0);
        if (top_down)
        {
            std::cout << "Deal with the negative height\n";
        }
        output.XSIZE = static_cast<unsigned int>(origin_x);
        output.YSIZE = static_cast<unsigned int>(std::abs(origin_y));
        std::printf("Width of the input image: %u\n", output.XSIZE);
        std::printf("Height of the input image: %u\n", output.YSIZE);
        std::printf("Size of the input image(Byte): %zu\n", static_cast<std::size_t>(output.XSIZE) * output.YSIZE * 3);
        output.FILLINGBYTE = bmp_filling_byte_calc(output.XSIZE);

        const std::size_t row_stride = static_cast<std::size_t>(output.XSIZE) * 3 + output.FILLINGBYTE;
        const std::size_t total_bytes = row_stride * output.YSIZE;
        output.IMAGE_DATA = static_cast<unsigned char*>(std::malloc(total_bytes));
        if (output.IMAGE_DATA == nullptr)
        {
            std::cerr << "Memory allocation error!";
            return output;
        }
        std::memset(output.IMAGE_DATA, 255, total_bytes);

        if (!top_down)
        {
            bmp_read_detail(output.IMAGE_DATA, static_cast<int>(output.XSIZE), static_cast<int>(output.YSIZE), output.FILENAME.string().c_str(), true);
            return output;
        }

        auto* const origin_data = static_cast<unsigned char*>(std::malloc(total_bytes));
        if (origin_data == nullptr)
        {
            std::cerr << "Memory allocation error!";
            return output;
        }
        std::memset(origin_data, 255, total_bytes);
        bmp_read_detail(origin_data, static_cast<int>(output.XSIZE), static_cast<int>(output.YSIZE), output.FILENAME.string().c_str(), true);

        for (unsigned int y = 0; y < output.YSIZE; ++y)
        {
            const std::size_t src_y = output.YSIZE - y - 1;
            std::memcpy(output.IMAGE_DATA + y * row_stride, origin_data + src_y * row_stride, row_stride);
        }
        std::free(origin_data);
        return output;
    }

    Image<RGB> bmp_read(const char* const filename, const bool extension)
    {
        auto image = bmp_file_read(filename, extension);
        auto output = TinyDIP::raw_image_to_array(static_cast<int>(image.XSIZE), static_cast<int>(image.YSIZE), image.IMAGE_DATA);
        std::free(image.IMAGE_DATA);
        return output;
    }

    Image<RGB> bmp_read(const std::string filename, const bool extension)
    {
        return bmp_read(filename.c_str(), extension);
    }

    int bmp_write(const std::string filename, Image<RGB> input)
    {
        return bmp_write(filename.c_str(), std::move(input));
    }

    int bmp_write(const char* const filename, Image<RGB> input)
    {
        return write_packed_image(
            filename,
            std::move(input),
            [](Image<RGB>&& img) { return TinyDIP::array_to_raw_image(std::move(img)); },
            [](const char* const f, const int sx, const int sy, const unsigned char* const raw) { return TinyDIP::bmp_write(f, sx, sy, raw); }
        );
    }

    int bmp_write(const char* const filename, const int xsize, const int ysize, const unsigned char* const image)
    {
        return write_bmp_container_file(filename, ".bmp", xsize, ysize, 3, 4, image);
    }

    unsigned char* array_to_raw_image(const Image<RGB> input)
    {
        return pack_3channel_bgr_buffer<RGB, unsigned char>(input, 4);
    }

    unsigned char bmp_filling_byte_calc(const unsigned int xsize, const int mod_num)
    {
        return static_cast<unsigned char>(xsize % static_cast<unsigned int>(mod_num));
    }

    namespace double_image
    {
        double* array_to_raw_image(const Image<double> input)
        {
            return pack_padded_row_buffer<double>(
                input,
                1,
                8,
                [](double* const dst, const double val) noexcept { *dst = val; }
            );
        }

        int write(const char* const filename, const int xsize, const int ysize, const double* const image)
        {
            return write_bmp_container_file(filename, ".dbmp", xsize, ysize, 1, 8, image);
        }

        int write(const char* const filename, Image<double> input)
        {
            return write_packed_image(
                filename,
                std::move(input),
                [](Image<double>&& img) { return TinyDIP::double_image::array_to_raw_image(std::move(img)); },
                [](const char* const f, const int sx, const int sy, const double* const raw) { return TinyDIP::double_image::write(f, sx, sy, raw); }
            );
        }

        TinyDIP::Image<double> read(const char* const filename, const bool extension)
        {
            return read_double_container_image<double>(
                filename,
                extension,
                ".dbmp",
                1,
                [](const int ox, const int oy, const double* const raw_buf)
                {
                    return unpack_padded_row_buffer<double>(
                        ox,
                        oy,
                        raw_buf,
                        1,
                        8,
                        [](double& dst, const double* const src) noexcept { dst = *src; }
                    );
                }
            );
        }

        double* array_to_raw_image(const Image<HSV> input)
        {
            return pack_3channel_bgr_buffer<HSV, double>(input, 8);
        }
    }

    int hsv_write_detail(const char* const filename, const int xsize, const int ysize, const double* const image)
    {
        return write_bmp_container_file(filename, ".hsv", xsize, ysize, 3, 8, image);
    }

    int hsv_write(const char* const filename, Image<HSV> input)
    {
        return write_packed_image(
            filename,
            std::move(input),
            [](Image<HSV>&& img) { return TinyDIP::double_image::array_to_raw_image(std::move(img)); },
            [](const char* const f, const int sx, const int sy, const double* const raw) { return TinyDIP::hsv_write_detail(f, sx, sy, raw); }
        );
    }

    Image<HSV> hsv_read(const char* const filename, const bool extension)
    {
        return read_double_container_image<HSV>(
            filename,
            extension,
            ".hsv",
            3,
            [](const int ox, const int oy, const double* const raw_buf)
            {
                return unpack_3channel_bgr_buffer<HSV>(ox, oy, raw_buf, 8);
            }
        );
    }
}
