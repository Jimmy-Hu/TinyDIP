/* Developed by Jimmy Hu */
/* Refactored for CLI Application capability */

//  compile command:
//  clang++ -std=c++20 -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp  main.cpp -L /usr/local/Cellar/llvm/10.0.0_3/lib/ -lm -O3 -o main -v
//  https://stackoverflow.com/a/61821729/6667035
//  clear && rm -rf ./main && g++-11 -std=c++20 -O4 -ffast-math -funsafe-math-optimizations -std=c++20 -fpermissive -H --verbose -Wall main.cpp -o main 


//#define USE_BOOST_ITERATOR
//#define USE_BOOST_SERIALIZATION

//  Standard Library Headers
#include <execution>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <valarray>
#include <vector>

//  Local Headers
#include "basic_functions.h"
#include "image_io.h"
#include "image_operations.h"
#include "timer.h"


//#define BOOST_TEST_DYN_LINK

//#define BOOST_TEST_MODULE image_elementwise_tests

#ifdef BOOST_TEST_MODULE
#include <boost/test/included/unit_test.hpp>


#ifdef BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#else
#include <boost/test/included/unit_test.hpp>
#endif // BOOST_TEST_DYN_LINK

#include <boost/mpl/list.hpp>
#include <boost/mpl/vector.hpp>
#include <tao/tuple/tuple.hpp>

typedef boost::mpl::list<
    byte, char, int, short, long, long long int,
    unsigned int, unsigned short int, unsigned long int, unsigned long long int,
    float, double, long double> test_types;

//  [TODO] Avoid code duplication (https://codereview.stackexchange.com/a/267709/231235)
BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_add_test, T, test_types)
{
    std::size_t size_x = 10;
    std::size_t size_y = 10;
    T initVal = 10;
    T increment = 1;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test += TinyDIP::Image<T>(size_x, size_y, increment);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal + increment));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_add_test_zero_dimensions, T, test_types)
{
    std::size_t size_x = 0;                         //  Test images with both of the dimensions having size zero.
    std::size_t size_y = 0;                         //  Test images with both of the dimensions having size zero.
    T initVal = 10;
    T increment = 1;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test += TinyDIP::Image<T>(size_x, size_y, increment);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal + increment));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_add_test_large_dimensions, T, test_types)
{
    std::size_t size_x = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    std::size_t size_y = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    T initVal = 10;
    T increment = 1;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test += TinyDIP::Image<T>(size_x, size_y, increment);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal + increment));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_minus_test, T, test_types)
{
    std::size_t size_x = 10;
    std::size_t size_y = 10;
    T initVal = 10;
    T difference = 1;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test -= TinyDIP::Image<T>(size_x, size_y, difference);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal - difference));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_minus_test_zero_dimensions, T, test_types)
{
    std::size_t size_x = 0;                         //  Test images with both of the dimensions having size zero.
    std::size_t size_y = 0;                         //  Test images with both of the dimensions having size zero.
    T initVal = 10;
    T difference = 1;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test -= TinyDIP::Image<T>(size_x, size_y, difference);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal - difference));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_minus_test_large_dimensions, T, test_types)
{
    std::size_t size_x = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    std::size_t size_y = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    T initVal = 10;
    T difference = 1;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test -= TinyDIP::Image<T>(size_x, size_y, difference);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal - difference));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_multiplies_test, T, test_types)
{
    std::size_t size_x = 10;
    std::size_t size_y = 10;
    T initVal = 10;
    T multiplier = 2;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test *= TinyDIP::Image<T>(size_x, size_y, multiplier);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal * multiplier));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_multiplies_test_zero_dimensions, T, test_types)
{
    std::size_t size_x = 0;                         //  Test images with both of the dimensions having size zero.
    std::size_t size_y = 0;                         //  Test images with both of the dimensions having size zero.
    T initVal = 10;
    T multiplier = 2;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test *= TinyDIP::Image<T>(size_x, size_y, multiplier);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal * multiplier));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_multiplies_test_large_dimensions, T, test_types)
{
    std::size_t size_x = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    std::size_t size_y = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    T initVal = 10;
    T multiplier = 2;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test *= TinyDIP::Image<T>(size_x, size_y, multiplier);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal * multiplier));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_divides_test, T, test_types)
{
    std::size_t size_x = 10;
    std::size_t size_y = 10;
    T initVal = 10;
    T divider = 2;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test /= TinyDIP::Image<T>(size_x, size_y, divider);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal / divider));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_divides_test_zero_dimensions, T, test_types)
{
    std::size_t size_x = 0;                         //  Test images with both of the dimensions having size zero.
    std::size_t size_y = 0;                         //  Test images with both of the dimensions having size zero.
    T initVal = 10;
    T divider = 2;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test /= TinyDIP::Image<T>(size_x, size_y, divider);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal / divider));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_divides_test_large_dimensions, T, test_types)
{
    std::size_t size_x = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    std::size_t size_y = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    T initVal = 10;
    T divider = 2;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test /= TinyDIP::Image<T>(size_x, size_y, divider);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal / divider));
}
/*
BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_divides_zero_test, T, test_types)
{
    std::size_t size_x = 10;
    std::size_t size_y = 10;
    T initVal = 10;
    T divider = 0;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test /= TinyDIP::Image<T>(size_x, size_y, divider);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal / divider));       //  dividing by zero test
}
*/
#endif

void difference_and_enhancement(std::string input_path1, std::string input_path2, double enhancement_times)
{
    if (input_path1.empty())
    {
        std::cerr << "Input path is empty!";
    }

    std::filesystem::path input1 = input_path1;
    std::filesystem::path input2 = input_path2;
    
}

#ifndef BOOST_TEST_MODULE
void addLeadingZeros(std::string input_path, std::string output_path);

//  parse_arg template function implementation
//  Helper for converting string to numeric types safely
template <typename T>
T parse_arg(const std::string& str)
{
    std::stringstream ss(str);
    T result;
    ss >> result;
    return result;
}

void print(auto comment, auto const& seq, char term = ' ') {
    for (std::cout << comment << '\n'; auto const& elem : seq)
        std::cout << elem << term;
    std::cout << '\n';
}


auto myHighLightRegion_parameters(const std::size_t index = 0)
{
    std::vector<std::tuple<
        std::string,    //  filenames
        std::size_t,    //  start_index
        std::size_t,    //  end_index
        std::size_t,    //  startx
        std::size_t,    //  endx
        std::size_t,    //  starty
        std::size_t,    //  endy
        std::string     //  output_location
        >> collection;
}

//  Command handler type definition
//  Uses std::function to store lambda or function pointers
using CommandHandler = std::function<void(const std::vector<std::string>& args)>;

//  CommandRegistry class implementation
class CommandRegistry
{
private:
    std::map<std::string, std::pair<std::string, CommandHandler>> commands;

public:
    void register_command(const std::string& name, const std::string& description, CommandHandler handler)
    {
        commands.emplace(name, std::make_pair(description, handler));
    }

    void list_commands() const
    {
        std::cout << "Available Commands:\n";
        for (const auto& [name, info] : commands)
        {
            std::cout << "  " << std::left << std::setw(15) << name << " : " << info.first << "\n";
        }
        std::cout << "\nUsage: ./tinydip <command> [args...]\n";
    }

    void execute(const std::string& command_name, const std::vector<std::string>& args)
    {
        if (commands.find(command_name) != commands.end())
        {
            try
            {
                // Execute the registered handler
                commands[command_name].second(args);
            }
            catch (const std::exception& e)
            {
                std::cerr << "Error executing command '" << command_name << "': " << e.what() << "\n";
            }
        }
        else
        {
            std::cerr << "Unknown command: " << command_name << "\n";
            list_commands();
        }
    }
};

//  ResizeHandler struct implementation
//  Wrapper for the 'resize' functionality
//  Args: input_path output_path width height
struct ResizeHandler
{
    void operator()(const std::vector<std::string>& args, std::ostream& os = std::cout) const
    {
        if (args.size() < 4)
        {
            std::cerr << "Usage: resize <input_bmp> <output_bmp> <width> <height>\n";
            return;
        }

        std::string input_path = args[0];
        std::string output_path = args[1];
        std::size_t width = parse_arg<std::size_t>(args[2]);
        std::size_t height = parse_arg<std::size_t>(args[3]);

        os << "Resizing " << input_path << " to " << width << "x" << height << "...\n";

        // Reading image
        auto input_img = TinyDIP::bmp_read(input_path.c_str(), true); // Assume true for convert to RGB/standard

        // Perform operation
        // Using execution policy if TinyDIP supports it internally, otherwise standard call
        auto output_img = TinyDIP::copyResizeBicubic(input_img, width, height);

        // Writing image
        TinyDIP::bmp_write(output_path.c_str(), output_img);
        os << "Saved to " << output_path << "\n";
    }
};

//  InfoHandler struct implementation
//  Wrapper for 'info' functionality
//  Args: input_path
struct InfoHandler
{
    void operator()(const std::vector<std::string>& args, std::ostream& os = std::cout) const
    {
        if (args.empty())
        {
            std::cerr << "Usage: info <input_bmp>\n";
            return;
        }

        std::string input_path = args[0];
        if (!std::filesystem::exists(input_path))
        {
            std::cerr << "File not found: " << input_path << "\n";
            return;
        }

        auto img = TinyDIP::bmp_read(input_path.c_str(), false);
        os << "Image Info:\n";
        os << "  Path:   " << input_path << "\n";
        os << "  Width:  " << img.getWidth() << "\n";
        os << "  Height: " << img.getHeight() << "\n";
        // Add more info if available (channels, etc.)
    }
};

int main()
{
    auto start = std::chrono::system_clock::now();
    std::string file_path = "InputImages/1";
    auto bmp1 = TinyDIP::bmp_read(file_path.c_str(), false);
    std::size_t N1 = 8, N2 = 8;
    auto block_count_x = bmp1.getWidth() / N1;
    auto block_count_y = bmp1.getHeight() / N2;

    bmp1 = TinyDIP::concat(TinyDIP::recursive_transform<2>(
        //std::execution::par,
        [](auto&& element)
        {
            auto hsv_block = TinyDIP::rgb2hsv(TinyDIP::im2double(element));
            auto v_block = TinyDIP::getVplane(hsv_block);
            auto v_block_dct = TinyDIP::dct2(v_block);
            return TinyDIP::hsv2rgb(TinyDIP::constructHSV(
                TinyDIP::getHplane(hsv_block),
                TinyDIP::getSplane(hsv_block),
                TinyDIP::idct2(v_block_dct)
            ));
        },
        TinyDIP::split(bmp1, block_count_x, block_count_y)));
    bmp1 = copyResizeBicubic(bmp1, bmp1.getWidth() * 2, bmp1.getHeight() * 2);
    //bmp1 = gaussian_fisheye(bmp1, 800.0);
    auto v_plane = TinyDIP::getVplane(TinyDIP::rgb2hsv(bmp1));
    auto SIFT_keypoints = TinyDIP::SIFT_impl::get_potential_keypoint(std::execution::par, v_plane);
    std::cout << "SIFT_keypoints = " << SIFT_keypoints.size() << "\n";
    bmp1 = TinyDIP::draw_points(bmp1, SIFT_keypoints);
    for (auto&& each_SIFT_keypoint : SIFT_keypoints)
    {
        TinyDIP::SIFT_impl::get_keypoint_descriptor(v_plane, each_SIFT_keypoint);
    }
    #ifdef USE_OPENCV
    auto cv_mat = TinyDIP::to_cv_mat(bmp1);
    cv::imshow("Image", cv_mat);
    cv::waitKey(0);
    #endif
    TinyDIP::bmp_write("test20241026", bmp1);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << " seconds\n";

    
    return EXIT_SUCCESS;
}

#endif


void test()
{
    constexpr int dims = 5;
    std::vector<std::string> test_vector1{ "1", "4", "7" };
    auto test1 = TinyDIP::n_dim_vector_generator<dims>(test_vector1, 3);
    std::vector<std::string> test_vector2{ "2", "5", "8" };
    auto test2 = TinyDIP::n_dim_vector_generator<dims>(test_vector2, 3);
    std::vector<std::string> test_vector3{ "3", "6", "9" };
    auto test3 = TinyDIP::n_dim_vector_generator<dims>(test_vector3, 3);
    std::vector<std::string> test_vector4{ "a", "b", "c" };
    auto test4 = TinyDIP::n_dim_vector_generator<dims>(test_vector4, 3);
    auto output = TinyDIP::recursive_transform<dims + 1>(
        [](auto element1, auto element2, auto element3, auto element4) { return element1 + element2 + element3 + element4; },
        test1, test2, test3, test4);
    std::cout << typeid(output).name() << std::endl;
    TinyDIP::recursive_print(output
    .at(0).at(0).at(0).at(0).at(0));
    return;   
}

//  bicubicInterpolationTest function implementation
void bicubicInterpolationTest()
{
    TinyDIP::Image<TinyDIP::GrayScale> image1(3, 3);
    image1.setAllValue(1);
    std::cout << "Width: " + std::to_string(image1.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image1.getHeight()) + "\n";
    image1.at(static_cast<std::size_t>(1), static_cast<std::size_t>(1)) = 100;
    image1.print();

    auto image2 = TinyDIP::copyResizeBicubic(image1, 12, 12);
    image2.print();
}

void addLeadingZeros(std::string input_path, std::string output_path)
{
    for (std::size_t i = 1; i <= 96; ++i)
    {
        std::string filename = input_path + std::to_string(i) + ".bmp";
        std::cout << filename << "\n";
        auto bmpimage = TinyDIP::bmp_read(filename.c_str(), true);
        char buff[100];
        snprintf(buff, sizeof(buff), "%s%05ld", output_path.c_str(), i);
        TinyDIP::bmp_write(buff, bmpimage);
    }
}
