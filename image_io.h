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

    namespace pnm
    {
        // -------------------------------------------------------------------------
        // read_pnm_token function implementation
        // Helper: Skip comments and read the next string token securely from a PPM
        // -------------------------------------------------------------------------
        inline std::string read_pnm_token(std::ifstream& file)
        {
            std::string token;
            while (file >> token)
            {
                if (token.empty())
                {
                    continue;
                }

                if (token[0] == '#')
                {
                    // If it's a comment, discard the rest of the line
                    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
                else
                {
                    return token;
                }
            }
            return "";
        }

        // -------------------------------------------------------------------------
        // shift_color_value function implementation
        // Helper: Precision bit shifting strictly ported from easyppm_read logic
        // -------------------------------------------------------------------------
        constexpr std::uint8_t shift_color_value(const int value, const int shift_bit)
        {
            if (shift_bit >= 0)
            {
                return static_cast<std::uint8_t>(value << shift_bit);
            }
            else
            {
                return static_cast<std::uint8_t>(value >> std::abs(shift_bit));
            }
        }

        // -------------------------------------------------------------------------
        // Default Lambda Object: Functor with operator() for pixel processing
        // -------------------------------------------------------------------------
        struct ProcessPPMData
        {
            const std::vector<int>& raw_data;
            const std::string& magic;
            const int shift_bit;
            const std::size_t width;
            const std::size_t height;
            TinyDIP::Image<RGB>& image;

            constexpr void operator()(const std::size_t i) const
            {
                const std::size_t x = i % width;
                const std::size_t y = i / width;
                const std::size_t flipped_y = height - 1 - y;
                
                RGB pixel{};

                if (magic == "P1")
                {
                    const int val = (raw_data[i] == 0) ? 1 : 0;
                    pixel.channels[0] = static_cast<std::uint8_t>(val);
                    pixel.channels[1] = static_cast<std::uint8_t>(val);
                    pixel.channels[2] = static_cast<std::uint8_t>(val);
                }
                else if (magic == "P2")
                {
                    const auto shifted_val = shift_color_value(raw_data[i], shift_bit);
                    pixel.channels[0] = shifted_val;
                    pixel.channels[1] = shifted_val;
                    pixel.channels[2] = shifted_val;
                }
                else if (magic == "P3")
                {
                    const std::size_t base_idx = i * 3;
                    pixel.channels[0] = shift_color_value(raw_data[base_idx], shift_bit);
                    pixel.channels[1] = shift_color_value(raw_data[base_idx + 1], shift_bit);
                    pixel.channels[2] = shift_color_value(raw_data[base_idx + 2], shift_bit);
                }

                image.at_without_boundary_check(x, flipped_y) = pixel;
            }
        };
        
        // -------------------------------------------------------------------------
        // transform_pixel_processing template function implementation
        // Generic Threading Engine for Data Processing (with invocable constraint)
        // -------------------------------------------------------------------------
        template <class ExecutionPolicy, typename Func, std::ranges::input_range RangeT>
        requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
                  std::invocable<Func, std::ranges::range_value_t<RangeT>>)
        void transform_pixel_processing(ExecutionPolicy&& policy, const RangeT& indices, Func&& func)
        {
            std::for_each(std::forward<ExecutionPolicy>(policy), std::ranges::begin(indices), std::ranges::end(indices), std::forward<Func>(func));
        }

        // transform_pixel_processing_omp template function implementation
        // Overload specifically designed to fall back to OpenMP
        template <typename Func, std::ranges::random_access_range RangeT>
        requires std::invocable<Func, std::ranges::range_value_t<RangeT>>
        void transform_pixel_processing_omp(const RangeT& indices, Func&& func)
        {
            #pragma omp parallel for
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(std::ranges::size(indices)); ++i)
            {
                func(indices[i]);
            }
        }

        // -------------------------------------------------------------------------
        // read template function implementation
        // Modern read_pnm incorporating execution policies and safety checks
        // -------------------------------------------------------------------------
        template <class ExecutionPolicy>
        requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
        TinyDIP::Image<RGB> read(ExecutionPolicy&& policy, const std::filesystem::path& file_path, const int outbits = 8)
        {
            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open())
            {
                throw std::runtime_error("Could not open file for reading: " + file_path.string());
            }

            const std::string magic = read_pnm_token(file);
            if (magic != "P1" && magic != "P2" && magic != "P3")
            {
                throw std::runtime_error("Unsupported image format. Magic number found: " + magic);
            }

            const std::size_t width = std::stoull(read_pnm_token(file));
            const std::size_t height = std::stoull(read_pnm_token(file));

            int max_value = 1; // Default for P1 (PBM)
            if (magic != "P1")
            {
                max_value = std::stoi(read_pnm_token(file));
            }

            const int in_bits = static_cast<int>(std::log2(max_value + 1));
            const int shift_bit = outbits - in_bits;

            const std::size_t pixel_count = width * height;
            const std::size_t expected_values = (magic == "P3") ? (pixel_count * 3) : pixel_count;

            // Fast-read phase securely ignoring comments throughout the matrix
            std::vector<int> raw_data;
            raw_data.reserve(expected_values);
            for (std::size_t i = 0; i < expected_values; ++i)
            {
                std::string token = read_pnm_token(file);
                if (token.empty())
                {
                    throw std::runtime_error("Unexpected end of file while reading PNM data.");
                }
                raw_data.emplace_back(std::stoi(token));
            }

            TinyDIP::Image<RGB> image(width, height);

            std::vector<std::size_t> indices(pixel_count);
            std::ranges::iota(indices, 0); // Using C++23 std::ranges::iota

            ProcessPPMData processor{ raw_data, magic, shift_bit, width, height, image };

            // Dispatches to execution policy with constraints verified
            transform_pixel_processing_omp(std::forward<ExecutionPolicy>(policy), indices, processor);

            return image;
        }

        // read template function implementation
        // Overload fallback avoiding execution policies, instead opting for OpenMP
        inline TinyDIP::Image<RGB> read(const std::filesystem::path& file_path, const int outbits = 8)
        {
            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open())
            {
                throw std::runtime_error("Could not open file for reading: " + file_path.string());
            }

            const std::string magic = read_pnm_token(file);
            if (magic != "P1" && magic != "P2" && magic != "P3")
            {
                throw std::runtime_error("Unsupported image format. Magic number found: " + magic);
            }

            const std::size_t width = std::stoull(read_pnm_token(file));
            const std::size_t height = std::stoull(read_pnm_token(file));

            int max_value = 1; 
            if (magic != "P1")
            {
                max_value = std::stoi(read_pnm_token(file));
            }

            const int in_bits = static_cast<int>(std::log2(max_value + 1));
            const int shift_bit = outbits - in_bits;

            const std::size_t pixel_count = width * height;
            const std::size_t expected_values = (magic == "P3") ? (pixel_count * 3) : pixel_count;

            std::vector<int> raw_data;
            raw_data.reserve(expected_values);
            for (std::size_t i = 0; i < expected_values; ++i)
            {
                std::string token = read_pnm_token(file);
                if (token.empty())
                {
                    throw std::runtime_error("Unexpected end of file while reading PNM data.");
                }
                raw_data.emplace_back(std::stoi(token));
            }

            TinyDIP::Image<RGB> image(width, height);

            std::vector<std::size_t> indices(pixel_count);
            std::ranges::iota(indices, 0); // Using C++23 std::ranges::iota

            ProcessPPMData processor{ raw_data, magic, shift_bit, width, height, image };

            // Dispatch explicitly to the OpenMP implementation
            transform_pixel_processing_omp(indices, processor);

            return image;
        }

        // -------------------------------------------------------------------------
        // write function implementation
        // This function writes an Image<RGB> to a PNM file with specified magic number
        // Modern write_pnm for exporting Image<RGB> to PNM files
        // -------------------------------------------------------------------------
        inline void write(
            const TinyDIP::Image<RGB>& image, 
            const std::filesystem::path& file_path, 
            const std::string& magic = "P3", 
            const int max_value = 255)
        {
            std::ofstream file(file_path, std::ios::binary);
            if (!file.is_open())
            {
                throw std::runtime_error("Could not open file for writing: " + file_path.string());
            }

            if (magic != "P1" && magic != "P2" && magic != "P3")
            {
                throw std::invalid_argument("Unsupported magic number for writing. Please use P1, P2, or P3.");
            }

            const std::size_t width = image.getWidth();
            const std::size_t height = image.getHeight();

            // Write header information
            file << magic << '\n';
            file << width << ' ' << height << '\n';
            if (magic != "P1")
            {
                file << max_value << '\n';
            }

            // Write pixel data sequentially mapped top-to-bottom
            for (std::size_t y = 0; y < height; ++y)
            {
                // Must flip y-axis back to match the standard PPM specification
                const std::size_t flipped_y = height - 1 - y;
                
                for (std::size_t x = 0; x < width; ++x)
                {
                    const RGB& pixel = image.at_without_boundary_check(x, flipped_y);
                    
                    if (magic == "P1")
                    {
                        // In P1, '1' traditionally corresponds to black pixel (0 intensity)
                        const int val = (pixel.channels[0] == 0) ? 1 : 0;
                        file << val << '\n';
                    }
                    else if (magic == "P2")
                    {
                        // In P2, write the first channel (grayscale)
                        file << +pixel.channels[0] << '\n';
                    }
                    else if (magic == "P3")
                    {
                        // In P3, write all three channels (RGB)
                        file << +pixel.channels[0] << ' ' 
                            << +pixel.channels[1] << ' ' 
                            << +pixel.channels[2] << '\n';
                    }
                }
            }
        }
    }

    int hsv_write_detail(const char* const filename, const int xsize, const int ysize, const double* const image);

    int hsv_write(const char* const filename, Image<HSV> input);

    Image<HSV> hsv_read(const char* const filename, const bool extension);
}

#endif

