//  Developed by Jimmy Hu

#include <cassert>
#include <execution>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../image_io.h"
#include "../cube.h"
#include "../cube_operations.h"
#include "../timer.h"

template<typename ElementT>
void print3(std::vector<TinyDIP::Image<ElementT>> input)
{
    for (std::size_t i = 0; i < input.size(); i++)
    {
        input[i].print();
        std::cout << "*******************\n";
    }
}

template<typename T>
void dct3LargeImageTest(std::size_t N = 100)
{
    std::size_t N1 = N, N2 = N, N3 = N;
    std::vector<TinyDIP::Image<T>> test_input;
    for (std::size_t z = 0; z < N3; z++)
    {
        test_input.push_back(TinyDIP::Image<T>(N1, N2));
    }
    for (std::size_t z = 1; z <= N3; z++)
    {
        for (std::size_t y = 1; y <= N2; y++)
        {
            for (std::size_t x = 1; x <= N1; x++)
            {
                if (std::fmod(z, 2) == 0)
                {
                    test_input[z - 1].at(y - 1, x - 1) = 255;
                }
                else
                {
                    test_input[z - 1].at(y - 1, x - 1) = 0;
                }
            }
        }
    }

    for (std::size_t i = 0; i < N; ++i)
    {
        std::string dir = "./DCT3/" + std::to_string(N) + "/";
        if (!std::filesystem::is_directory(dir))
        {
            std::filesystem::create_directories(dir);
        }
        std::filesystem::path filename = std::string(dir + std::to_string(i) + std::string(".dbmp"));
        if (!std::filesystem::exists(filename))
        {
            std::cout << "Calculate " << std::to_string(N) << "x" << std::to_string(N) << "x"<< std::to_string(N) << " DCT, plane " << std::to_string(i) << '\n'; 
            auto test_output = TinyDIP::dct3_one_plane(test_input, i);
            std::filesystem::path path_without_extension = filename.parent_path() / filename.stem();
            TinyDIP::double_image::write(path_without_extension.string().c_str(), test_output);
        }
    }
}

int main()
{
    TinyDIP::Timer timer1;
    dct3LargeImageTest<double>();
    return EXIT_SUCCESS;
}