#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

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
void dct3DetailTest()
{
    std::size_t N1 = 10, N2 = 10, N3 = 10;
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
                test_input[z - 1].at(y - 1, x - 1) = x * 100 + y * 10 + z;
            }
        }
    }
    print3(test_input);

    auto test_output = TinyDIP::dct3_one_plane(test_input, 0);
    test_output.print();
}

int main()
{
    auto start = std::chrono::system_clock::now();
    dct3DetailTest<float>();
    dct3DetailTest<double>();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}