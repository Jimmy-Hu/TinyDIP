//  recursiveTransformTypeTest
//  Developed by Jimmy Hu

#include <algorithm>
#include <cassert>
#include <execution>
#include <filesystem>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../image_io.h"
#include "../cube.h"
#include "../cube_operations.h"
#include "../timer.h"

// Define a type alias to capture the container, function, expected type, and the actual resulting type.
// Notice that the Function parameter now precedes the Container parameter in the decltype evaluation 
// to match the updated TinyDIP::recursive_transform signature.
template <std::size_t unwrap_level, typename Container, typename Function, typename Expected>
using return_type_test_type = std::tuple<
    Container,
    Function,
    Expected,
    decltype(TinyDIP::recursive_transform<unwrap_level>(std::execution::par, std::declval<Function>(), std::declval<Container>()))
>;

// The collection of types to test.
using return_type_test_types = std::tuple<
    //                    Container   Function                Result

    // Non-range. (unwrap_level is 0 for all scalar types)
    //  template
    //  Replace "\n//T1 replace finished.\r\n\r\n" into "\n"
    /*
    return_type_test_type<0, T1,        T2 (*)(T1),        T2>,
    return_type_test_type<0, T1,        T2 (*)(T1 const&), T2>,
    return_type_test_type<0, T1 const,  T2 (*)(T1 const&), T2>,
    return_type_test_type<0, T1&,       T2 (*)(T1 const&), T2>,
    return_type_test_type<0, T1 const&, T2 (*)(T1 const&), T2>,
    return_type_test_type<0, T1&&,      T2 (*)(T1 const&), T2>,
    */
    return_type_test_type<0, char,        char (*)(char),        char>,
    return_type_test_type<0, char,        char (*)(char const&), char>,
    return_type_test_type<0, char const,  char (*)(char const&), char>,
    return_type_test_type<0, char&,       char (*)(char const&), char>,
    return_type_test_type<0, char const&, char (*)(char const&), char>,
    return_type_test_type<0, char&&,      char (*)(char const&), char>,

    return_type_test_type<0, int,        char (*)(int),        char>,
    return_type_test_type<0, int,        char (*)(int const&), char>,
    return_type_test_type<0, int const,  char (*)(int const&), char>,
    return_type_test_type<0, int&,       char (*)(int const&), char>,
    return_type_test_type<0, int const&, char (*)(int const&), char>,
    return_type_test_type<0, int&&,      char (*)(int const&), char>,

    return_type_test_type<0, short,        char (*)(short),        char>,
    return_type_test_type<0, short,        char (*)(short const&), char>,
    return_type_test_type<0, short const,  char (*)(short const&), char>,
    return_type_test_type<0, short&,       char (*)(short const&), char>,
    return_type_test_type<0, short const&, char (*)(short const&), char>,
    return_type_test_type<0, short&&,      char (*)(short const&), char>,

    return_type_test_type<0, long,        char (*)(long),        char>,
    return_type_test_type<0, long,        char (*)(long const&), char>,
    return_type_test_type<0, long const,  char (*)(long const&), char>,
    return_type_test_type<0, long&,       char (*)(long const&), char>,
    return_type_test_type<0, long const&, char (*)(long const&), char>,
    return_type_test_type<0, long&&,      char (*)(long const&), char>,

    return_type_test_type<0, long long int,        char (*)(long long int),        char>,
    return_type_test_type<0, long long int,        char (*)(long long int const&), char>,
    return_type_test_type<0, long long int const,  char (*)(long long int const&), char>,
    return_type_test_type<0, long long int&,       char (*)(long long int const&), char>,
    return_type_test_type<0, long long int const&, char (*)(long long int const&), char>,
    return_type_test_type<0, long long int&&,      char (*)(long long int const&), char>,

    return_type_test_type<0, unsigned char,        char (*)(unsigned char),        char>,
    return_type_test_type<0, unsigned char,        char (*)(unsigned char const&), char>,
    return_type_test_type<0, unsigned char const,  char (*)(unsigned char const&), char>,
    return_type_test_type<0, unsigned char&,       char (*)(unsigned char const&), char>,
    return_type_test_type<0, unsigned char const&, char (*)(unsigned char const&), char>,
    return_type_test_type<0, unsigned char&&,      char (*)(unsigned char const&), char>,

    return_type_test_type<0, unsigned int,        char (*)(unsigned int),        char>,
    return_type_test_type<0, unsigned int,        char (*)(unsigned int const&), char>,
    return_type_test_type<0, unsigned int const,  char (*)(unsigned int const&), char>,
    return_type_test_type<0, unsigned int&,       char (*)(unsigned int const&), char>,
    return_type_test_type<0, unsigned int const&, char (*)(unsigned int const&), char>,
    return_type_test_type<0, unsigned int&&,      char (*)(unsigned int const&), char>,

    return_type_test_type<0, unsigned short int,        char (*)(unsigned short int),        char>,
    return_type_test_type<0, unsigned short int,        char (*)(unsigned short int const&), char>,
    return_type_test_type<0, unsigned short int const,  char (*)(unsigned short int const&), char>,
    return_type_test_type<0, unsigned short int&,       char (*)(unsigned short int const&), char>,
    return_type_test_type<0, unsigned short int const&, char (*)(unsigned short int const&), char>,
    return_type_test_type<0, unsigned short int&&,      char (*)(unsigned short int const&), char>,

    return_type_test_type<0, unsigned long int,        char (*)(unsigned long int),        char>,
    return_type_test_type<0, unsigned long int,        char (*)(unsigned long int const&), char>,
    return_type_test_type<0, unsigned long int const,  char (*)(unsigned long int const&), char>,
    return_type_test_type<0, unsigned long int&,       char (*)(unsigned long int const&), char>,
    return_type_test_type<0, unsigned long int const&, char (*)(unsigned long int const&), char>,
    return_type_test_type<0, unsigned long int&&,      char (*)(unsigned long int const&), char>,

    return_type_test_type<0, float,        char (*)(float),        char>,
    return_type_test_type<0, float,        char (*)(float const&), char>,
    return_type_test_type<0, float const,  char (*)(float const&), char>,
    return_type_test_type<0, float&,       char (*)(float const&), char>,
    return_type_test_type<0, float const&, char (*)(float const&), char>,
    return_type_test_type<0, float&&,      char (*)(float const&), char>,

    return_type_test_type<0, double,        char (*)(double),        char>,
    return_type_test_type<0, double,        char (*)(double const&), char>,
    return_type_test_type<0, double const,  char (*)(double const&), char>,
    return_type_test_type<0, double&,       char (*)(double const&), char>,
    return_type_test_type<0, double const&, char (*)(double const&), char>,
    return_type_test_type<0, double&&,      char (*)(double const&), char>,

    return_type_test_type<0, long double,        char (*)(long double),        char>,
    return_type_test_type<0, long double,        char (*)(long double const&), char>,
    return_type_test_type<0, long double const,  char (*)(long double const&), char>,
    return_type_test_type<0, long double&,       char (*)(long double const&), char>,
    return_type_test_type<0, long double const&, char (*)(long double const&), char>,
    return_type_test_type<0, long double&&,      char (*)(long double const&), char>,

    return_type_test_type<0, std::string,        char (*)(std::string),        char>,
    return_type_test_type<0, std::string,        char (*)(std::string const&), char>,
    return_type_test_type<0, std::string const,  char (*)(std::string const&), char>,
    return_type_test_type<0, std::string&,       char (*)(std::string const&), char>,
    return_type_test_type<0, std::string const&, char (*)(std::string const&), char>,
    return_type_test_type<0, std::string&&,      char (*)(std::string const&), char>,

    return_type_test_type<0, std::complex<char>,        char (*)(std::complex<char>),        char>,
    return_type_test_type<0, std::complex<char>,        char (*)(std::complex<char> const&), char>,
    return_type_test_type<0, std::complex<char> const,  char (*)(std::complex<char> const&), char>,
    return_type_test_type<0, std::complex<char>&,       char (*)(std::complex<char> const&), char>,
    return_type_test_type<0, std::complex<char> const&, char (*)(std::complex<char> const&), char>,
    return_type_test_type<0, std::complex<char>&&,      char (*)(std::complex<char> const&), char>,

    return_type_test_type<0, std::complex<int>,        char (*)(std::complex<int>),        char>,
    return_type_test_type<0, std::complex<int>,        char (*)(std::complex<int> const&), char>,
    return_type_test_type<0, std::complex<int> const,  char (*)(std::complex<int> const&), char>,
    return_type_test_type<0, std::complex<int>&,       char (*)(std::complex<int> const&), char>,
    return_type_test_type<0, std::complex<int> const&, char (*)(std::complex<int> const&), char>,
    return_type_test_type<0, std::complex<int>&&,      char (*)(std::complex<int> const&), char>,

    return_type_test_type<0, std::complex<short>,        char (*)(std::complex<short>),        char>,
    return_type_test_type<0, std::complex<short>,        char (*)(std::complex<short> const&), char>,
    return_type_test_type<0, std::complex<short> const,  char (*)(std::complex<short> const&), char>,
    return_type_test_type<0, std::complex<short>&,       char (*)(std::complex<short> const&), char>,
    return_type_test_type<0, std::complex<short> const&, char (*)(std::complex<short> const&), char>,
    return_type_test_type<0, std::complex<short>&&,      char (*)(std::complex<short> const&), char>,

    return_type_test_type<0, std::complex<long>,        char (*)(std::complex<long>),        char>,
    return_type_test_type<0, std::complex<long>,        char (*)(std::complex<long> const&), char>,
    return_type_test_type<0, std::complex<long> const,  char (*)(std::complex<long> const&), char>,
    return_type_test_type<0, std::complex<long>&,       char (*)(std::complex<long> const&), char>,
    return_type_test_type<0, std::complex<long> const&, char (*)(std::complex<long> const&), char>,
    return_type_test_type<0, std::complex<long>&&,      char (*)(std::complex<long> const&), char>,

    return_type_test_type<0, std::complex<long long int>,        char (*)(std::complex<long long int>),        char>,
    return_type_test_type<0, std::complex<long long int>,        char (*)(std::complex<long long int> const&), char>,
    return_type_test_type<0, std::complex<long long int> const,  char (*)(std::complex<long long int> const&), char>,
    return_type_test_type<0, std::complex<long long int>&,       char (*)(std::complex<long long int> const&), char>,
    return_type_test_type<0, std::complex<long long int> const&, char (*)(std::complex<long long int> const&), char>,
    return_type_test_type<0, std::complex<long long int>&&,      char (*)(std::complex<long long int> const&), char>,

    return_type_test_type<0, std::complex<unsigned char>,        char (*)(std::complex<unsigned char>),        char>,
    return_type_test_type<0, std::complex<unsigned char>,        char (*)(std::complex<unsigned char> const&), char>,
    return_type_test_type<0, std::complex<unsigned char> const,  char (*)(std::complex<unsigned char> const&), char>,
    return_type_test_type<0, std::complex<unsigned char>&,       char (*)(std::complex<unsigned char> const&), char>,
    return_type_test_type<0, std::complex<unsigned char> const&, char (*)(std::complex<unsigned char> const&), char>,
    return_type_test_type<0, std::complex<unsigned char>&&,      char (*)(std::complex<unsigned char> const&), char>,

    return_type_test_type<0, std::complex<unsigned int>,        char (*)(std::complex<unsigned int>),        char>,
    return_type_test_type<0, std::complex<unsigned int>,        char (*)(std::complex<unsigned int> const&), char>,
    return_type_test_type<0, std::complex<unsigned int> const,  char (*)(std::complex<unsigned int> const&), char>,
    return_type_test_type<0, std::complex<unsigned int>&,       char (*)(std::complex<unsigned int> const&), char>,
    return_type_test_type<0, std::complex<unsigned int> const&, char (*)(std::complex<unsigned int> const&), char>,
    return_type_test_type<0, std::complex<unsigned int>&&,      char (*)(std::complex<unsigned int> const&), char>,

    return_type_test_type<0, std::complex<unsigned short int>,        char (*)(std::complex<unsigned short int>),        char>,
    return_type_test_type<0, std::complex<unsigned short int>,        char (*)(std::complex<unsigned short int> const&), char>,
    return_type_test_type<0, std::complex<unsigned short int> const,  char (*)(std::complex<unsigned short int> const&), char>,
    return_type_test_type<0, std::complex<unsigned short int>&,       char (*)(std::complex<unsigned short int> const&), char>,
    return_type_test_type<0, std::complex<unsigned short int> const&, char (*)(std::complex<unsigned short int> const&), char>,
    return_type_test_type<0, std::complex<unsigned short int>&&,      char (*)(std::complex<unsigned short int> const&), char>,

    return_type_test_type<0, std::complex<unsigned long int>,        char (*)(std::complex<unsigned long int>),        char>,
    return_type_test_type<0, std::complex<unsigned long int>,        char (*)(std::complex<unsigned long int> const&), char>,
    return_type_test_type<0, std::complex<unsigned long int> const,  char (*)(std::complex<unsigned long int> const&), char>,
    return_type_test_type<0, std::complex<unsigned long int>&,       char (*)(std::complex<unsigned long int> const&), char>,
    return_type_test_type<0, std::complex<unsigned long int> const&, char (*)(std::complex<unsigned long int> const&), char>,
    return_type_test_type<0, std::complex<unsigned long int>&&,      char (*)(std::complex<unsigned long int> const&), char>,

    return_type_test_type<0, std::complex<float>,        char (*)(std::complex<float>),        char>,
    return_type_test_type<0, std::complex<float>,        char (*)(std::complex<float> const&), char>,
    return_type_test_type<0, std::complex<float> const,  char (*)(std::complex<float> const&), char>,
    return_type_test_type<0, std::complex<float>&,       char (*)(std::complex<float> const&), char>,
    return_type_test_type<0, std::complex<float> const&, char (*)(std::complex<float> const&), char>,
    return_type_test_type<0, std::complex<float>&&,      char (*)(std::complex<float> const&), char>,

    return_type_test_type<0, std::complex<double>,        char (*)(std::complex<double>),        char>,
    return_type_test_type<0, std::complex<double>,        char (*)(std::complex<double> const&), char>,
    return_type_test_type<0, std::complex<double> const,  char (*)(std::complex<double> const&), char>,
    return_type_test_type<0, std::complex<double>&,       char (*)(std::complex<double> const&), char>,
    return_type_test_type<0, std::complex<double> const&, char (*)(std::complex<double> const&), char>,
    return_type_test_type<0, std::complex<double>&&,      char (*)(std::complex<double> const&), char>,

    return_type_test_type<0, std::complex<long double>,        char (*)(std::complex<long double>),        char>,
    return_type_test_type<0, std::complex<long double>,        char (*)(std::complex<long double> const&), char>,
    return_type_test_type<0, std::complex<long double> const,  char (*)(std::complex<long double> const&), char>,
    return_type_test_type<0, std::complex<long double>&,       char (*)(std::complex<long double> const&), char>,
    return_type_test_type<0, std::complex<long double> const&, char (*)(std::complex<long double> const&), char>,
    return_type_test_type<0, std::complex<long double>&&,      char (*)(std::complex<long double> const&), char>,

    return_type_test_type<0, std::optional<char>,        char (*)(std::optional<char>),        char>,
    return_type_test_type<0, std::optional<char>,        char (*)(std::optional<char> const&), char>,
    return_type_test_type<0, std::optional<char> const,  char (*)(std::optional<char> const&), char>,
    return_type_test_type<0, std::optional<char>&,       char (*)(std::optional<char> const&), char>,
    return_type_test_type<0, std::optional<char> const&, char (*)(std::optional<char> const&), char>,
    return_type_test_type<0, std::optional<char>&&,      char (*)(std::optional<char> const&), char>,

    return_type_test_type<0, std::optional<int>,        char (*)(std::optional<int>),        char>,
    return_type_test_type<0, std::optional<int>,        char (*)(std::optional<int> const&), char>,
    return_type_test_type<0, std::optional<int> const,  char (*)(std::optional<int> const&), char>,
    return_type_test_type<0, std::optional<int>&,       char (*)(std::optional<int> const&), char>,
    return_type_test_type<0, std::optional<int> const&, char (*)(std::optional<int> const&), char>,
    return_type_test_type<0, std::optional<int>&&,      char (*)(std::optional<int> const&), char>,

    return_type_test_type<0, std::optional<short>,        char (*)(std::optional<short>),        char>,
    return_type_test_type<0, std::optional<short>,        char (*)(std::optional<short> const&), char>,
    return_type_test_type<0, std::optional<short> const,  char (*)(std::optional<short> const&), char>,
    return_type_test_type<0, std::optional<short>&,       char (*)(std::optional<short> const&), char>,
    return_type_test_type<0, std::optional<short> const&, char (*)(std::optional<short> const&), char>,
    return_type_test_type<0, std::optional<short>&&,      char (*)(std::optional<short> const&), char>,

    return_type_test_type<0, std::optional<long>,        char (*)(std::optional<long>),        char>,
    return_type_test_type<0, std::optional<long>,        char (*)(std::optional<long> const&), char>,
    return_type_test_type<0, std::optional<long> const,  char (*)(std::optional<long> const&), char>,
    return_type_test_type<0, std::optional<long>&,       char (*)(std::optional<long> const&), char>,
    return_type_test_type<0, std::optional<long> const&, char (*)(std::optional<long> const&), char>,
    return_type_test_type<0, std::optional<long>&&,      char (*)(std::optional<long> const&), char>,

    return_type_test_type<0, std::optional<long long int>,        char (*)(std::optional<long long int>),        char>,
    return_type_test_type<0, std::optional<long long int>,        char (*)(std::optional<long long int> const&), char>,
    return_type_test_type<0, std::optional<long long int> const,  char (*)(std::optional<long long int> const&), char>,
    return_type_test_type<0, std::optional<long long int>&,       char (*)(std::optional<long long int> const&), char>,
    return_type_test_type<0, std::optional<long long int> const&, char (*)(std::optional<long long int> const&), char>,
    return_type_test_type<0, std::optional<long long int>&&,      char (*)(std::optional<long long int> const&), char>,

    return_type_test_type<0, std::optional<unsigned char>,        char (*)(std::optional<unsigned char>),        char>,
    return_type_test_type<0, std::optional<unsigned char>,        char (*)(std::optional<unsigned char> const&), char>,
    return_type_test_type<0, std::optional<unsigned char> const,  char (*)(std::optional<unsigned char> const&), char>,
    return_type_test_type<0, std::optional<unsigned char>&,       char (*)(std::optional<unsigned char> const&), char>,
    return_type_test_type<0, std::optional<unsigned char> const&, char (*)(std::optional<unsigned char> const&), char>,
    return_type_test_type<0, std::optional<unsigned char>&&,      char (*)(std::optional<unsigned char> const&), char>,

    return_type_test_type<0, std::optional<unsigned int>,        char (*)(std::optional<unsigned int>),        char>,
    return_type_test_type<0, std::optional<unsigned int>,        char (*)(std::optional<unsigned int> const&), char>,
    return_type_test_type<0, std::optional<unsigned int> const,  char (*)(std::optional<unsigned int> const&), char>,
    return_type_test_type<0, std::optional<unsigned int>&,       char (*)(std::optional<unsigned int> const&), char>,
    return_type_test_type<0, std::optional<unsigned int> const&, char (*)(std::optional<unsigned int> const&), char>,
    return_type_test_type<0, std::optional<unsigned int>&&,      char (*)(std::optional<unsigned int> const&), char>,

    return_type_test_type<0, std::optional<unsigned short int>,        char (*)(std::optional<unsigned short int>),        char>,
    return_type_test_type<0, std::optional<unsigned short int>,        char (*)(std::optional<unsigned short int> const&), char>,
    return_type_test_type<0, std::optional<unsigned short int> const,  char (*)(std::optional<unsigned short int> const&), char>,
    return_type_test_type<0, std::optional<unsigned short int>&,       char (*)(std::optional<unsigned short int> const&), char>,
    return_type_test_type<0, std::optional<unsigned short int> const&, char (*)(std::optional<unsigned short int> const&), char>,
    return_type_test_type<0, std::optional<unsigned short int>&&,      char (*)(std::optional<unsigned short int> const&), char>,

    return_type_test_type<0, std::optional<unsigned long int>,        char (*)(std::optional<unsigned long int>),        char>,
    return_type_test_type<0, std::optional<unsigned long int>,        char (*)(std::optional<unsigned long int> const&), char>,
    return_type_test_type<0, std::optional<unsigned long int> const,  char (*)(std::optional<unsigned long int> const&), char>,
    return_type_test_type<0, std::optional<unsigned long int>&,       char (*)(std::optional<unsigned long int> const&), char>,
    return_type_test_type<0, std::optional<unsigned long int> const&, char (*)(std::optional<unsigned long int> const&), char>,
    return_type_test_type<0, std::optional<unsigned long int>&&,      char (*)(std::optional<unsigned long int> const&), char>,

    return_type_test_type<0, std::optional<float>,        char (*)(std::optional<float>),        char>,
    return_type_test_type<0, std::optional<float>,        char (*)(std::optional<float> const&), char>,
    return_type_test_type<0, std::optional<float> const,  char (*)(std::optional<float> const&), char>,
    return_type_test_type<0, std::optional<float>&,       char (*)(std::optional<float> const&), char>,
    return_type_test_type<0, std::optional<float> const&, char (*)(std::optional<float> const&), char>,
    return_type_test_type<0, std::optional<float>&&,      char (*)(std::optional<float> const&), char>,

    return_type_test_type<0, std::optional<double>,        char (*)(std::optional<double>),        char>,
    return_type_test_type<0, std::optional<double>,        char (*)(std::optional<double> const&), char>,
    return_type_test_type<0, std::optional<double> const,  char (*)(std::optional<double> const&), char>,
    return_type_test_type<0, std::optional<double>&,       char (*)(std::optional<double> const&), char>,
    return_type_test_type<0, std::optional<double> const&, char (*)(std::optional<double> const&), char>,
    return_type_test_type<0, std::optional<double>&&,      char (*)(std::optional<double> const&), char>,

    return_type_test_type<0, std::optional<long double>,        char (*)(std::optional<long double>),        char>,
    return_type_test_type<0, std::optional<long double>,        char (*)(std::optional<long double> const&), char>,
    return_type_test_type<0, std::optional<long double> const,  char (*)(std::optional<long double> const&), char>,
    return_type_test_type<0, std::optional<long double>&,       char (*)(std::optional<long double> const&), char>,
    return_type_test_type<0, std::optional<long double> const&, char (*)(std::optional<long double> const&), char>,
    return_type_test_type<0, std::optional<long double>&&,      char (*)(std::optional<long double> const&), char>,

    return_type_test_type<0, std::optional<std::string>,        char (*)(std::optional<std::string>),        char>,
    return_type_test_type<0, std::optional<std::string>,        char (*)(std::optional<std::string> const&), char>,
    return_type_test_type<0, std::optional<std::string> const,  char (*)(std::optional<std::string> const&), char>,
    return_type_test_type<0, std::optional<std::string>&,       char (*)(std::optional<std::string> const&), char>,
    return_type_test_type<0, std::optional<std::string> const&, char (*)(std::optional<std::string> const&), char>,
    return_type_test_type<0, std::optional<std::string>&&,      char (*)(std::optional<std::string> const&), char>,

    return_type_test_type<0, std::optional<std::complex<char>>,        char (*)(std::optional<std::complex<char>>),        char>,
    return_type_test_type<0, std::optional<std::complex<char>>,        char (*)(std::optional<std::complex<char>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<char>> const,  char (*)(std::optional<std::complex<char>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<char>>&,       char (*)(std::optional<std::complex<char>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<char>> const&, char (*)(std::optional<std::complex<char>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<char>>&&,      char (*)(std::optional<std::complex<char>> const&), char>,

    return_type_test_type<0, std::optional<std::complex<int>>,        char (*)(std::optional<std::complex<int>>),        char>,
    return_type_test_type<0, std::optional<std::complex<int>>,        char (*)(std::optional<std::complex<int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<int>> const,  char (*)(std::optional<std::complex<int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<int>>&,       char (*)(std::optional<std::complex<int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<int>> const&, char (*)(std::optional<std::complex<int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<int>>&&,      char (*)(std::optional<std::complex<int>> const&), char>,

    return_type_test_type<0, std::optional<std::complex<short>>,        char (*)(std::optional<std::complex<short>>),        char>,
    return_type_test_type<0, std::optional<std::complex<short>>,        char (*)(std::optional<std::complex<short>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<short>> const,  char (*)(std::optional<std::complex<short>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<short>>&,       char (*)(std::optional<std::complex<short>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<short>> const&, char (*)(std::optional<std::complex<short>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<short>>&&,      char (*)(std::optional<std::complex<short>> const&), char>,

    return_type_test_type<0, std::optional<std::complex<long>>,        char (*)(std::optional<std::complex<long>>),        char>,
    return_type_test_type<0, std::optional<std::complex<long>>,        char (*)(std::optional<std::complex<long>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<long>> const,  char (*)(std::optional<std::complex<long>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<long>>&,       char (*)(std::optional<std::complex<long>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<long>> const&, char (*)(std::optional<std::complex<long>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<long>>&&,      char (*)(std::optional<std::complex<long>> const&), char>,

    return_type_test_type<0, std::optional<std::complex<long long int>>,        char (*)(std::optional<std::complex<long long int>>),        char>,
    return_type_test_type<0, std::optional<std::complex<long long int>>,        char (*)(std::optional<std::complex<long long int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<long long int>> const,  char (*)(std::optional<std::complex<long long int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<long long int>>&,       char (*)(std::optional<std::complex<long long int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<long long int>> const&, char (*)(std::optional<std::complex<long long int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<long long int>>&&,      char (*)(std::optional<std::complex<long long int>> const&), char>,

    return_type_test_type<0, std::optional<std::complex<unsigned char>>,        char (*)(std::optional<std::complex<unsigned char>>),        char>,
    return_type_test_type<0, std::optional<std::complex<unsigned char>>,        char (*)(std::optional<std::complex<unsigned char>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned char>> const,  char (*)(std::optional<std::complex<unsigned char>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned char>>&,       char (*)(std::optional<std::complex<unsigned char>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned char>> const&, char (*)(std::optional<std::complex<unsigned char>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned char>>&&,      char (*)(std::optional<std::complex<unsigned char>> const&), char>,

    return_type_test_type<0, std::optional<std::complex<unsigned int>>,        char (*)(std::optional<std::complex<unsigned int>>),        char>,
    return_type_test_type<0, std::optional<std::complex<unsigned int>>,        char (*)(std::optional<std::complex<unsigned int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned int>> const,  char (*)(std::optional<std::complex<unsigned int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned int>>&,       char (*)(std::optional<std::complex<unsigned int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned int>> const&, char (*)(std::optional<std::complex<unsigned int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned int>>&&,      char (*)(std::optional<std::complex<unsigned int>> const&), char>,

    return_type_test_type<0, std::optional<std::complex<unsigned short int>>,        char (*)(std::optional<std::complex<unsigned short int>>),        char>,
    return_type_test_type<0, std::optional<std::complex<unsigned short int>>,        char (*)(std::optional<std::complex<unsigned short int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned short int>> const,  char (*)(std::optional<std::complex<unsigned short int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned short int>>&,       char (*)(std::optional<std::complex<unsigned short int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned short int>> const&, char (*)(std::optional<std::complex<unsigned short int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned short int>>&&,      char (*)(std::optional<std::complex<unsigned short int>> const&), char>,

    return_type_test_type<0, std::optional<std::complex<unsigned long int>>,        char (*)(std::optional<std::complex<unsigned long int>>),        char>,
    return_type_test_type<0, std::optional<std::complex<unsigned long int>>,        char (*)(std::optional<std::complex<unsigned long int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned long int>> const,  char (*)(std::optional<std::complex<unsigned long int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned long int>>&,       char (*)(std::optional<std::complex<unsigned long int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned long int>> const&, char (*)(std::optional<std::complex<unsigned long int>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<unsigned long int>>&&,      char (*)(std::optional<std::complex<unsigned long int>> const&), char>,

    return_type_test_type<0, std::optional<std::complex<float>>,        char (*)(std::optional<std::complex<float>>),        char>,
    return_type_test_type<0, std::optional<std::complex<float>>,        char (*)(std::optional<std::complex<float>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<float>> const,  char (*)(std::optional<std::complex<float>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<float>>&,       char (*)(std::optional<std::complex<float>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<float>> const&, char (*)(std::optional<std::complex<float>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<float>>&&,      char (*)(std::optional<std::complex<float>> const&), char>,

    return_type_test_type<0, std::optional<std::complex<double>>,        char (*)(std::optional<std::complex<double>>),        char>,
    return_type_test_type<0, std::optional<std::complex<double>>,        char (*)(std::optional<std::complex<double>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<double>> const,  char (*)(std::optional<std::complex<double>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<double>>&,       char (*)(std::optional<std::complex<double>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<double>> const&, char (*)(std::optional<std::complex<double>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<double>>&&,      char (*)(std::optional<std::complex<double>> const&), char>,

    return_type_test_type<0, std::optional<std::complex<long double>>,        char (*)(std::optional<std::complex<long double>>),        char>,
    return_type_test_type<0, std::optional<std::complex<long double>>,        char (*)(std::optional<std::complex<long double>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<long double>> const,  char (*)(std::optional<std::complex<long double>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<long double>>&,       char (*)(std::optional<std::complex<long double>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<long double>> const&, char (*)(std::optional<std::complex<long double>> const&), char>,
    return_type_test_type<0, std::optional<std::complex<long double>>&&,      char (*)(std::optional<std::complex<long double>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<char>>,        char (*)(std::optional<std::optional<char>>),        char>,
    return_type_test_type<0, std::optional<std::optional<char>>,        char (*)(std::optional<std::optional<char>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<char>> const,  char (*)(std::optional<std::optional<char>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<char>>&,       char (*)(std::optional<std::optional<char>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<char>> const&, char (*)(std::optional<std::optional<char>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<char>>&&,      char (*)(std::optional<std::optional<char>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<int>>,        char (*)(std::optional<std::optional<int>>),        char>,
    return_type_test_type<0, std::optional<std::optional<int>>,        char (*)(std::optional<std::optional<int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<int>> const,  char (*)(std::optional<std::optional<int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<int>>&,       char (*)(std::optional<std::optional<int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<int>> const&, char (*)(std::optional<std::optional<int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<int>>&&,      char (*)(std::optional<std::optional<int>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<short>>,        char (*)(std::optional<std::optional<short>>),        char>,
    return_type_test_type<0, std::optional<std::optional<short>>,        char (*)(std::optional<std::optional<short>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<short>> const,  char (*)(std::optional<std::optional<short>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<short>>&,       char (*)(std::optional<std::optional<short>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<short>> const&, char (*)(std::optional<std::optional<short>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<short>>&&,      char (*)(std::optional<std::optional<short>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<long>>,        char (*)(std::optional<std::optional<long>>),        char>,
    return_type_test_type<0, std::optional<std::optional<long>>,        char (*)(std::optional<std::optional<long>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<long>> const,  char (*)(std::optional<std::optional<long>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<long>>&,       char (*)(std::optional<std::optional<long>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<long>> const&, char (*)(std::optional<std::optional<long>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<long>>&&,      char (*)(std::optional<std::optional<long>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<long long int>>,        char (*)(std::optional<std::optional<long long int>>),        char>,
    return_type_test_type<0, std::optional<std::optional<long long int>>,        char (*)(std::optional<std::optional<long long int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<long long int>> const,  char (*)(std::optional<std::optional<long long int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<long long int>>&,       char (*)(std::optional<std::optional<long long int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<long long int>> const&, char (*)(std::optional<std::optional<long long int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<long long int>>&&,      char (*)(std::optional<std::optional<long long int>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<unsigned char>>,        char (*)(std::optional<std::optional<unsigned char>>),        char>,
    return_type_test_type<0, std::optional<std::optional<unsigned char>>,        char (*)(std::optional<std::optional<unsigned char>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned char>> const,  char (*)(std::optional<std::optional<unsigned char>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned char>>&,       char (*)(std::optional<std::optional<unsigned char>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned char>> const&, char (*)(std::optional<std::optional<unsigned char>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned char>>&&,      char (*)(std::optional<std::optional<unsigned char>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<unsigned int>>,        char (*)(std::optional<std::optional<unsigned int>>),        char>,
    return_type_test_type<0, std::optional<std::optional<unsigned int>>,        char (*)(std::optional<std::optional<unsigned int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned int>> const,  char (*)(std::optional<std::optional<unsigned int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned int>>&,       char (*)(std::optional<std::optional<unsigned int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned int>> const&, char (*)(std::optional<std::optional<unsigned int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned int>>&&,      char (*)(std::optional<std::optional<unsigned int>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<unsigned short int>>,        char (*)(std::optional<std::optional<unsigned short int>>),        char>,
    return_type_test_type<0, std::optional<std::optional<unsigned short int>>,        char (*)(std::optional<std::optional<unsigned short int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned short int>> const,  char (*)(std::optional<std::optional<unsigned short int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned short int>>&,       char (*)(std::optional<std::optional<unsigned short int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned short int>> const&, char (*)(std::optional<std::optional<unsigned short int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned short int>>&&,      char (*)(std::optional<std::optional<unsigned short int>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<unsigned long int>>,        char (*)(std::optional<std::optional<unsigned long int>>),        char>,
    return_type_test_type<0, std::optional<std::optional<unsigned long int>>,        char (*)(std::optional<std::optional<unsigned long int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned long int>> const,  char (*)(std::optional<std::optional<unsigned long int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned long int>>&,       char (*)(std::optional<std::optional<unsigned long int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned long int>> const&, char (*)(std::optional<std::optional<unsigned long int>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<unsigned long int>>&&,      char (*)(std::optional<std::optional<unsigned long int>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<float>>,        char (*)(std::optional<std::optional<float>>),        char>,
    return_type_test_type<0, std::optional<std::optional<float>>,        char (*)(std::optional<std::optional<float>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<float>> const,  char (*)(std::optional<std::optional<float>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<float>>&,       char (*)(std::optional<std::optional<float>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<float>> const&, char (*)(std::optional<std::optional<float>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<float>>&&,      char (*)(std::optional<std::optional<float>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<double>>,        char (*)(std::optional<std::optional<double>>),        char>,
    return_type_test_type<0, std::optional<std::optional<double>>,        char (*)(std::optional<std::optional<double>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<double>> const,  char (*)(std::optional<std::optional<double>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<double>>&,       char (*)(std::optional<std::optional<double>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<double>> const&, char (*)(std::optional<std::optional<double>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<double>>&&,      char (*)(std::optional<std::optional<double>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<long double>>,        char (*)(std::optional<std::optional<long double>>),        char>,
    return_type_test_type<0, std::optional<std::optional<long double>>,        char (*)(std::optional<std::optional<long double>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<long double>> const,  char (*)(std::optional<std::optional<long double>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<long double>>&,       char (*)(std::optional<std::optional<long double>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<long double>> const&, char (*)(std::optional<std::optional<long double>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<long double>>&&,      char (*)(std::optional<std::optional<long double>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<std::string>>,        char (*)(std::optional<std::optional<std::string>>),        char>,
    return_type_test_type<0, std::optional<std::optional<std::string>>,        char (*)(std::optional<std::optional<std::string>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::string>> const,  char (*)(std::optional<std::optional<std::string>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::string>>&,       char (*)(std::optional<std::optional<std::string>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::string>> const&, char (*)(std::optional<std::optional<std::string>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::string>>&&,      char (*)(std::optional<std::optional<std::string>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<std::complex<char>>>,        char (*)(std::optional<std::optional<std::complex<char>>>),        char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<char>>>,        char (*)(std::optional<std::optional<std::complex<char>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<char>>> const,  char (*)(std::optional<std::optional<std::complex<char>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<char>>>&,       char (*)(std::optional<std::optional<std::complex<char>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<char>>> const&, char (*)(std::optional<std::optional<std::complex<char>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<char>>>&&,      char (*)(std::optional<std::optional<std::complex<char>>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<std::complex<int>>>,        char (*)(std::optional<std::optional<std::complex<int>>>),        char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<int>>>,        char (*)(std::optional<std::optional<std::complex<int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<int>>> const,  char (*)(std::optional<std::optional<std::complex<int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<int>>>&,       char (*)(std::optional<std::optional<std::complex<int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<int>>> const&, char (*)(std::optional<std::optional<std::complex<int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<int>>>&&,      char (*)(std::optional<std::optional<std::complex<int>>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<std::complex<short>>>,        char (*)(std::optional<std::optional<std::complex<short>>>),        char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<short>>>,        char (*)(std::optional<std::optional<std::complex<short>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<short>>> const,  char (*)(std::optional<std::optional<std::complex<short>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<short>>>&,       char (*)(std::optional<std::optional<std::complex<short>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<short>>> const&, char (*)(std::optional<std::optional<std::complex<short>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<short>>>&&,      char (*)(std::optional<std::optional<std::complex<short>>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<std::complex<long>>>,        char (*)(std::optional<std::optional<std::complex<long>>>),        char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long>>>,        char (*)(std::optional<std::optional<std::complex<long>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long>>> const,  char (*)(std::optional<std::optional<std::complex<long>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long>>>&,       char (*)(std::optional<std::optional<std::complex<long>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long>>> const&, char (*)(std::optional<std::optional<std::complex<long>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long>>>&&,      char (*)(std::optional<std::optional<std::complex<long>>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<std::complex<long long int>>>,        char (*)(std::optional<std::optional<std::complex<long long int>>>),        char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long long int>>>,        char (*)(std::optional<std::optional<std::complex<long long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long long int>>> const,  char (*)(std::optional<std::optional<std::complex<long long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long long int>>>&,       char (*)(std::optional<std::optional<std::complex<long long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long long int>>> const&, char (*)(std::optional<std::optional<std::complex<long long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long long int>>>&&,      char (*)(std::optional<std::optional<std::complex<long long int>>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned char>>>,        char (*)(std::optional<std::optional<std::complex<unsigned char>>>),        char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned char>>>,        char (*)(std::optional<std::optional<std::complex<unsigned char>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned char>>> const,  char (*)(std::optional<std::optional<std::complex<unsigned char>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned char>>>&,       char (*)(std::optional<std::optional<std::complex<unsigned char>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned char>>> const&, char (*)(std::optional<std::optional<std::complex<unsigned char>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned char>>>&&,      char (*)(std::optional<std::optional<std::complex<unsigned char>>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned int>>>,        char (*)(std::optional<std::optional<std::complex<unsigned int>>>),        char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned int>>>,        char (*)(std::optional<std::optional<std::complex<unsigned int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned int>>> const,  char (*)(std::optional<std::optional<std::complex<unsigned int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned int>>>&,       char (*)(std::optional<std::optional<std::complex<unsigned int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned int>>> const&, char (*)(std::optional<std::optional<std::complex<unsigned int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned int>>>&&,      char (*)(std::optional<std::optional<std::complex<unsigned int>>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned short int>>>,        char (*)(std::optional<std::optional<std::complex<unsigned short int>>>),        char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned short int>>>,        char (*)(std::optional<std::optional<std::complex<unsigned short int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned short int>>> const,  char (*)(std::optional<std::optional<std::complex<unsigned short int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned short int>>>&,       char (*)(std::optional<std::optional<std::complex<unsigned short int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned short int>>> const&, char (*)(std::optional<std::optional<std::complex<unsigned short int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned short int>>>&&,      char (*)(std::optional<std::optional<std::complex<unsigned short int>>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned long int>>>,        char (*)(std::optional<std::optional<std::complex<unsigned long int>>>),        char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned long int>>>,        char (*)(std::optional<std::optional<std::complex<unsigned long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned long int>>> const,  char (*)(std::optional<std::optional<std::complex<unsigned long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned long int>>>&,       char (*)(std::optional<std::optional<std::complex<unsigned long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned long int>>> const&, char (*)(std::optional<std::optional<std::complex<unsigned long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<unsigned long int>>>&&,      char (*)(std::optional<std::optional<std::complex<unsigned long int>>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<std::complex<float>>>,        char (*)(std::optional<std::optional<std::complex<float>>>),        char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<float>>>,        char (*)(std::optional<std::optional<std::complex<float>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<float>>> const,  char (*)(std::optional<std::optional<std::complex<float>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<float>>>&,       char (*)(std::optional<std::optional<std::complex<float>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<float>>> const&, char (*)(std::optional<std::optional<std::complex<float>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<float>>>&&,      char (*)(std::optional<std::optional<std::complex<float>>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<std::complex<double>>>,        char (*)(std::optional<std::optional<std::complex<double>>>),        char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<double>>>,        char (*)(std::optional<std::optional<std::complex<double>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<double>>> const,  char (*)(std::optional<std::optional<std::complex<double>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<double>>>&,       char (*)(std::optional<std::optional<std::complex<double>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<double>>> const&, char (*)(std::optional<std::optional<std::complex<double>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<double>>>&&,      char (*)(std::optional<std::optional<std::complex<double>>> const&), char>,

    return_type_test_type<0, std::optional<std::optional<std::complex<long double>>>,        char (*)(std::optional<std::optional<std::complex<long double>>>),        char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long double>>>,        char (*)(std::optional<std::optional<std::complex<long double>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long double>>> const,  char (*)(std::optional<std::optional<std::complex<long double>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long double>>>&,       char (*)(std::optional<std::optional<std::complex<long double>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long double>>> const&, char (*)(std::optional<std::optional<std::complex<long double>>> const&), char>,
    return_type_test_type<0, std::optional<std::optional<std::complex<long double>>>&&,      char (*)(std::optional<std::optional<std::complex<long double>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<char>>,        char (*)(std::optional<std::deque<char>>),        char>,
    return_type_test_type<0, std::optional<std::deque<char>>,        char (*)(std::optional<std::deque<char>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<char>> const,  char (*)(std::optional<std::deque<char>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<char>>&,       char (*)(std::optional<std::deque<char>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<char>> const&, char (*)(std::optional<std::deque<char>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<char>>&&,      char (*)(std::optional<std::deque<char>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<int>>,        char (*)(std::optional<std::deque<int>>),        char>,
    return_type_test_type<0, std::optional<std::deque<int>>,        char (*)(std::optional<std::deque<int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<int>> const,  char (*)(std::optional<std::deque<int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<int>>&,       char (*)(std::optional<std::deque<int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<int>> const&, char (*)(std::optional<std::deque<int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<int>>&&,      char (*)(std::optional<std::deque<int>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<short>>,        char (*)(std::optional<std::deque<short>>),        char>,
    return_type_test_type<0, std::optional<std::deque<short>>,        char (*)(std::optional<std::deque<short>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<short>> const,  char (*)(std::optional<std::deque<short>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<short>>&,       char (*)(std::optional<std::deque<short>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<short>> const&, char (*)(std::optional<std::deque<short>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<short>>&&,      char (*)(std::optional<std::deque<short>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<long>>,        char (*)(std::optional<std::deque<long>>),        char>,
    return_type_test_type<0, std::optional<std::deque<long>>,        char (*)(std::optional<std::deque<long>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<long>> const,  char (*)(std::optional<std::deque<long>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<long>>&,       char (*)(std::optional<std::deque<long>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<long>> const&, char (*)(std::optional<std::deque<long>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<long>>&&,      char (*)(std::optional<std::deque<long>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<long long int>>,        char (*)(std::optional<std::deque<long long int>>),        char>,
    return_type_test_type<0, std::optional<std::deque<long long int>>,        char (*)(std::optional<std::deque<long long int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<long long int>> const,  char (*)(std::optional<std::deque<long long int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<long long int>>&,       char (*)(std::optional<std::deque<long long int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<long long int>> const&, char (*)(std::optional<std::deque<long long int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<long long int>>&&,      char (*)(std::optional<std::deque<long long int>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<unsigned char>>,        char (*)(std::optional<std::deque<unsigned char>>),        char>,
    return_type_test_type<0, std::optional<std::deque<unsigned char>>,        char (*)(std::optional<std::deque<unsigned char>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned char>> const,  char (*)(std::optional<std::deque<unsigned char>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned char>>&,       char (*)(std::optional<std::deque<unsigned char>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned char>> const&, char (*)(std::optional<std::deque<unsigned char>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned char>>&&,      char (*)(std::optional<std::deque<unsigned char>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<unsigned int>>,        char (*)(std::optional<std::deque<unsigned int>>),        char>,
    return_type_test_type<0, std::optional<std::deque<unsigned int>>,        char (*)(std::optional<std::deque<unsigned int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned int>> const,  char (*)(std::optional<std::deque<unsigned int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned int>>&,       char (*)(std::optional<std::deque<unsigned int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned int>> const&, char (*)(std::optional<std::deque<unsigned int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned int>>&&,      char (*)(std::optional<std::deque<unsigned int>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<unsigned short int>>,        char (*)(std::optional<std::deque<unsigned short int>>),        char>,
    return_type_test_type<0, std::optional<std::deque<unsigned short int>>,        char (*)(std::optional<std::deque<unsigned short int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned short int>> const,  char (*)(std::optional<std::deque<unsigned short int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned short int>>&,       char (*)(std::optional<std::deque<unsigned short int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned short int>> const&, char (*)(std::optional<std::deque<unsigned short int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned short int>>&&,      char (*)(std::optional<std::deque<unsigned short int>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<unsigned long int>>,        char (*)(std::optional<std::deque<unsigned long int>>),        char>,
    return_type_test_type<0, std::optional<std::deque<unsigned long int>>,        char (*)(std::optional<std::deque<unsigned long int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned long int>> const,  char (*)(std::optional<std::deque<unsigned long int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned long int>>&,       char (*)(std::optional<std::deque<unsigned long int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned long int>> const&, char (*)(std::optional<std::deque<unsigned long int>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<unsigned long int>>&&,      char (*)(std::optional<std::deque<unsigned long int>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<float>>,        char (*)(std::optional<std::deque<float>>),        char>,
    return_type_test_type<0, std::optional<std::deque<float>>,        char (*)(std::optional<std::deque<float>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<float>> const,  char (*)(std::optional<std::deque<float>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<float>>&,       char (*)(std::optional<std::deque<float>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<float>> const&, char (*)(std::optional<std::deque<float>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<float>>&&,      char (*)(std::optional<std::deque<float>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<double>>,        char (*)(std::optional<std::deque<double>>),        char>,
    return_type_test_type<0, std::optional<std::deque<double>>,        char (*)(std::optional<std::deque<double>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<double>> const,  char (*)(std::optional<std::deque<double>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<double>>&,       char (*)(std::optional<std::deque<double>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<double>> const&, char (*)(std::optional<std::deque<double>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<double>>&&,      char (*)(std::optional<std::deque<double>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<long double>>,        char (*)(std::optional<std::deque<long double>>),        char>,
    return_type_test_type<0, std::optional<std::deque<long double>>,        char (*)(std::optional<std::deque<long double>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<long double>> const,  char (*)(std::optional<std::deque<long double>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<long double>>&,       char (*)(std::optional<std::deque<long double>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<long double>> const&, char (*)(std::optional<std::deque<long double>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<long double>>&&,      char (*)(std::optional<std::deque<long double>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::string>>,        char (*)(std::optional<std::deque<std::string>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::string>>,        char (*)(std::optional<std::deque<std::string>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::string>> const,  char (*)(std::optional<std::deque<std::string>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::string>>&,       char (*)(std::optional<std::deque<std::string>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::string>> const&, char (*)(std::optional<std::deque<std::string>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::string>>&&,      char (*)(std::optional<std::deque<std::string>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::complex<char>>>,        char (*)(std::optional<std::deque<std::complex<char>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<char>>>,        char (*)(std::optional<std::deque<std::complex<char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<char>>> const,  char (*)(std::optional<std::deque<std::complex<char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<char>>>&,       char (*)(std::optional<std::deque<std::complex<char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<char>>> const&, char (*)(std::optional<std::deque<std::complex<char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<char>>>&&,      char (*)(std::optional<std::deque<std::complex<char>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::complex<int>>>,        char (*)(std::optional<std::deque<std::complex<int>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<int>>>,        char (*)(std::optional<std::deque<std::complex<int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<int>>> const,  char (*)(std::optional<std::deque<std::complex<int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<int>>>&,       char (*)(std::optional<std::deque<std::complex<int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<int>>> const&, char (*)(std::optional<std::deque<std::complex<int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<int>>>&&,      char (*)(std::optional<std::deque<std::complex<int>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::complex<short>>>,        char (*)(std::optional<std::deque<std::complex<short>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<short>>>,        char (*)(std::optional<std::deque<std::complex<short>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<short>>> const,  char (*)(std::optional<std::deque<std::complex<short>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<short>>>&,       char (*)(std::optional<std::deque<std::complex<short>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<short>>> const&, char (*)(std::optional<std::deque<std::complex<short>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<short>>>&&,      char (*)(std::optional<std::deque<std::complex<short>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::complex<long>>>,        char (*)(std::optional<std::deque<std::complex<long>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long>>>,        char (*)(std::optional<std::deque<std::complex<long>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long>>> const,  char (*)(std::optional<std::deque<std::complex<long>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long>>>&,       char (*)(std::optional<std::deque<std::complex<long>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long>>> const&, char (*)(std::optional<std::deque<std::complex<long>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long>>>&&,      char (*)(std::optional<std::deque<std::complex<long>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::complex<long long int>>>,        char (*)(std::optional<std::deque<std::complex<long long int>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long long int>>>,        char (*)(std::optional<std::deque<std::complex<long long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long long int>>> const,  char (*)(std::optional<std::deque<std::complex<long long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long long int>>>&,       char (*)(std::optional<std::deque<std::complex<long long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long long int>>> const&, char (*)(std::optional<std::deque<std::complex<long long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long long int>>>&&,      char (*)(std::optional<std::deque<std::complex<long long int>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned char>>>,        char (*)(std::optional<std::deque<std::complex<unsigned char>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned char>>>,        char (*)(std::optional<std::deque<std::complex<unsigned char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned char>>> const,  char (*)(std::optional<std::deque<std::complex<unsigned char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned char>>>&,       char (*)(std::optional<std::deque<std::complex<unsigned char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned char>>> const&, char (*)(std::optional<std::deque<std::complex<unsigned char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned char>>>&&,      char (*)(std::optional<std::deque<std::complex<unsigned char>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned int>>>,        char (*)(std::optional<std::deque<std::complex<unsigned int>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned int>>>,        char (*)(std::optional<std::deque<std::complex<unsigned int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned int>>> const,  char (*)(std::optional<std::deque<std::complex<unsigned int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned int>>>&,       char (*)(std::optional<std::deque<std::complex<unsigned int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned int>>> const&, char (*)(std::optional<std::deque<std::complex<unsigned int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned int>>>&&,      char (*)(std::optional<std::deque<std::complex<unsigned int>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned short int>>>,        char (*)(std::optional<std::deque<std::complex<unsigned short int>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned short int>>>,        char (*)(std::optional<std::deque<std::complex<unsigned short int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned short int>>> const,  char (*)(std::optional<std::deque<std::complex<unsigned short int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned short int>>>&,       char (*)(std::optional<std::deque<std::complex<unsigned short int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned short int>>> const&, char (*)(std::optional<std::deque<std::complex<unsigned short int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned short int>>>&&,      char (*)(std::optional<std::deque<std::complex<unsigned short int>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned long int>>>,        char (*)(std::optional<std::deque<std::complex<unsigned long int>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned long int>>>,        char (*)(std::optional<std::deque<std::complex<unsigned long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned long int>>> const,  char (*)(std::optional<std::deque<std::complex<unsigned long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned long int>>>&,       char (*)(std::optional<std::deque<std::complex<unsigned long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned long int>>> const&, char (*)(std::optional<std::deque<std::complex<unsigned long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<unsigned long int>>>&&,      char (*)(std::optional<std::deque<std::complex<unsigned long int>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::complex<float>>>,        char (*)(std::optional<std::deque<std::complex<float>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<float>>>,        char (*)(std::optional<std::deque<std::complex<float>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<float>>> const,  char (*)(std::optional<std::deque<std::complex<float>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<float>>>&,       char (*)(std::optional<std::deque<std::complex<float>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<float>>> const&, char (*)(std::optional<std::deque<std::complex<float>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<float>>>&&,      char (*)(std::optional<std::deque<std::complex<float>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::complex<double>>>,        char (*)(std::optional<std::deque<std::complex<double>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<double>>>,        char (*)(std::optional<std::deque<std::complex<double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<double>>> const,  char (*)(std::optional<std::deque<std::complex<double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<double>>>&,       char (*)(std::optional<std::deque<std::complex<double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<double>>> const&, char (*)(std::optional<std::deque<std::complex<double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<double>>>&&,      char (*)(std::optional<std::deque<std::complex<double>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::complex<long double>>>,        char (*)(std::optional<std::deque<std::complex<long double>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long double>>>,        char (*)(std::optional<std::deque<std::complex<long double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long double>>> const,  char (*)(std::optional<std::deque<std::complex<long double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long double>>>&,       char (*)(std::optional<std::deque<std::complex<long double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long double>>> const&, char (*)(std::optional<std::deque<std::complex<long double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::complex<long double>>>&&,      char (*)(std::optional<std::deque<std::complex<long double>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<char>>>,        char (*)(std::optional<std::deque<std::optional<char>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<char>>>,        char (*)(std::optional<std::deque<std::optional<char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<char>>> const,  char (*)(std::optional<std::deque<std::optional<char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<char>>>&,       char (*)(std::optional<std::deque<std::optional<char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<char>>> const&, char (*)(std::optional<std::deque<std::optional<char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<char>>>&&,      char (*)(std::optional<std::deque<std::optional<char>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<int>>>,        char (*)(std::optional<std::deque<std::optional<int>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<int>>>,        char (*)(std::optional<std::deque<std::optional<int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<int>>> const,  char (*)(std::optional<std::deque<std::optional<int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<int>>>&,       char (*)(std::optional<std::deque<std::optional<int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<int>>> const&, char (*)(std::optional<std::deque<std::optional<int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<int>>>&&,      char (*)(std::optional<std::deque<std::optional<int>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<short>>>,        char (*)(std::optional<std::deque<std::optional<short>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<short>>>,        char (*)(std::optional<std::deque<std::optional<short>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<short>>> const,  char (*)(std::optional<std::deque<std::optional<short>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<short>>>&,       char (*)(std::optional<std::deque<std::optional<short>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<short>>> const&, char (*)(std::optional<std::deque<std::optional<short>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<short>>>&&,      char (*)(std::optional<std::deque<std::optional<short>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<long>>>,        char (*)(std::optional<std::deque<std::optional<long>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long>>>,        char (*)(std::optional<std::deque<std::optional<long>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long>>> const,  char (*)(std::optional<std::deque<std::optional<long>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long>>>&,       char (*)(std::optional<std::deque<std::optional<long>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long>>> const&, char (*)(std::optional<std::deque<std::optional<long>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long>>>&&,      char (*)(std::optional<std::deque<std::optional<long>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<long long int>>>,        char (*)(std::optional<std::deque<std::optional<long long int>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long long int>>>,        char (*)(std::optional<std::deque<std::optional<long long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long long int>>> const,  char (*)(std::optional<std::deque<std::optional<long long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long long int>>>&,       char (*)(std::optional<std::deque<std::optional<long long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long long int>>> const&, char (*)(std::optional<std::deque<std::optional<long long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long long int>>>&&,      char (*)(std::optional<std::deque<std::optional<long long int>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned char>>>,        char (*)(std::optional<std::deque<std::optional<unsigned char>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned char>>>,        char (*)(std::optional<std::deque<std::optional<unsigned char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned char>>> const,  char (*)(std::optional<std::deque<std::optional<unsigned char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned char>>>&,       char (*)(std::optional<std::deque<std::optional<unsigned char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned char>>> const&, char (*)(std::optional<std::deque<std::optional<unsigned char>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned char>>>&&,      char (*)(std::optional<std::deque<std::optional<unsigned char>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned int>>>,        char (*)(std::optional<std::deque<std::optional<unsigned int>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned int>>>,        char (*)(std::optional<std::deque<std::optional<unsigned int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned int>>> const,  char (*)(std::optional<std::deque<std::optional<unsigned int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned int>>>&,       char (*)(std::optional<std::deque<std::optional<unsigned int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned int>>> const&, char (*)(std::optional<std::deque<std::optional<unsigned int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned int>>>&&,      char (*)(std::optional<std::deque<std::optional<unsigned int>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned short int>>>,        char (*)(std::optional<std::deque<std::optional<unsigned short int>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned short int>>>,        char (*)(std::optional<std::deque<std::optional<unsigned short int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned short int>>> const,  char (*)(std::optional<std::deque<std::optional<unsigned short int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned short int>>>&,       char (*)(std::optional<std::deque<std::optional<unsigned short int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned short int>>> const&, char (*)(std::optional<std::deque<std::optional<unsigned short int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned short int>>>&&,      char (*)(std::optional<std::deque<std::optional<unsigned short int>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned long int>>>,        char (*)(std::optional<std::deque<std::optional<unsigned long int>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned long int>>>,        char (*)(std::optional<std::deque<std::optional<unsigned long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned long int>>> const,  char (*)(std::optional<std::deque<std::optional<unsigned long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned long int>>>&,       char (*)(std::optional<std::deque<std::optional<unsigned long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned long int>>> const&, char (*)(std::optional<std::deque<std::optional<unsigned long int>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<unsigned long int>>>&&,      char (*)(std::optional<std::deque<std::optional<unsigned long int>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<float>>>,        char (*)(std::optional<std::deque<std::optional<float>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<float>>>,        char (*)(std::optional<std::deque<std::optional<float>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<float>>> const,  char (*)(std::optional<std::deque<std::optional<float>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<float>>>&,       char (*)(std::optional<std::deque<std::optional<float>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<float>>> const&, char (*)(std::optional<std::deque<std::optional<float>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<float>>>&&,      char (*)(std::optional<std::deque<std::optional<float>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<double>>>,        char (*)(std::optional<std::deque<std::optional<double>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<double>>>,        char (*)(std::optional<std::deque<std::optional<double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<double>>> const,  char (*)(std::optional<std::deque<std::optional<double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<double>>>&,       char (*)(std::optional<std::deque<std::optional<double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<double>>> const&, char (*)(std::optional<std::deque<std::optional<double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<double>>>&&,      char (*)(std::optional<std::deque<std::optional<double>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<long double>>>,        char (*)(std::optional<std::deque<std::optional<long double>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long double>>>,        char (*)(std::optional<std::deque<std::optional<long double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long double>>> const,  char (*)(std::optional<std::deque<std::optional<long double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long double>>>&,       char (*)(std::optional<std::deque<std::optional<long double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long double>>> const&, char (*)(std::optional<std::deque<std::optional<long double>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<long double>>>&&,      char (*)(std::optional<std::deque<std::optional<long double>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<std::string>>>,        char (*)(std::optional<std::deque<std::optional<std::string>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::string>>>,        char (*)(std::optional<std::deque<std::optional<std::string>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::string>>> const,  char (*)(std::optional<std::deque<std::optional<std::string>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::string>>>&,       char (*)(std::optional<std::deque<std::optional<std::string>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::string>>> const&, char (*)(std::optional<std::deque<std::optional<std::string>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::string>>>&&,      char (*)(std::optional<std::deque<std::optional<std::string>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<char>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<char>>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<char>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<char>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<char>>>> const,  char (*)(std::optional<std::deque<std::optional<std::complex<char>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<char>>>>&,       char (*)(std::optional<std::deque<std::optional<std::complex<char>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<char>>>> const&, char (*)(std::optional<std::deque<std::optional<std::complex<char>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<char>>>>&&,      char (*)(std::optional<std::deque<std::optional<std::complex<char>>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<int>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<int>>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<int>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<int>>>> const,  char (*)(std::optional<std::deque<std::optional<std::complex<int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<int>>>>&,       char (*)(std::optional<std::deque<std::optional<std::complex<int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<int>>>> const&, char (*)(std::optional<std::deque<std::optional<std::complex<int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<int>>>>&&,      char (*)(std::optional<std::deque<std::optional<std::complex<int>>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<short>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<short>>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<short>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<short>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<short>>>> const,  char (*)(std::optional<std::deque<std::optional<std::complex<short>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<short>>>>&,       char (*)(std::optional<std::deque<std::optional<std::complex<short>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<short>>>> const&, char (*)(std::optional<std::deque<std::optional<std::complex<short>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<short>>>>&&,      char (*)(std::optional<std::deque<std::optional<std::complex<short>>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<long>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<long>>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<long>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<long>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<long>>>> const,  char (*)(std::optional<std::deque<std::optional<std::complex<long>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<long>>>>&,       char (*)(std::optional<std::deque<std::optional<std::complex<long>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<long>>>> const&, char (*)(std::optional<std::deque<std::optional<std::complex<long>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<long>>>>&&,      char (*)(std::optional<std::deque<std::optional<std::complex<long>>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<long long int>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<long long int>>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<long long int>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<long long int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<long long int>>>> const,  char (*)(std::optional<std::deque<std::optional<std::complex<long long int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<long long int>>>>&,       char (*)(std::optional<std::deque<std::optional<std::complex<long long int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<long long int>>>> const&, char (*)(std::optional<std::deque<std::optional<std::complex<long long int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<long long int>>>>&&,      char (*)(std::optional<std::deque<std::optional<std::complex<long long int>>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned char>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<unsigned char>>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned char>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<unsigned char>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned char>>>> const,  char (*)(std::optional<std::deque<std::optional<std::complex<unsigned char>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned char>>>>&,       char (*)(std::optional<std::deque<std::optional<std::complex<unsigned char>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned char>>>> const&, char (*)(std::optional<std::deque<std::optional<std::complex<unsigned char>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned char>>>>&&,      char (*)(std::optional<std::deque<std::optional<std::complex<unsigned char>>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned int>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<unsigned int>>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned int>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<unsigned int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned int>>>> const,  char (*)(std::optional<std::deque<std::optional<std::complex<unsigned int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned int>>>>&,       char (*)(std::optional<std::deque<std::optional<std::complex<unsigned int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned int>>>> const&, char (*)(std::optional<std::deque<std::optional<std::complex<unsigned int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned int>>>>&&,      char (*)(std::optional<std::deque<std::optional<std::complex<unsigned int>>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned short int>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<unsigned short int>>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned short int>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<unsigned short int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned short int>>>> const,  char (*)(std::optional<std::deque<std::optional<std::complex<unsigned short int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned short int>>>>&,       char (*)(std::optional<std::deque<std::optional<std::complex<unsigned short int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned short int>>>> const&, char (*)(std::optional<std::deque<std::optional<std::complex<unsigned short int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned short int>>>>&&,      char (*)(std::optional<std::deque<std::optional<std::complex<unsigned short int>>>> const&), char>,

    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned long int>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<unsigned long int>>>>),        char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned long int>>>>,        char (*)(std::optional<std::deque<std::optional<std::complex<unsigned long int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned long int>>>> const,  char (*)(std::optional<std::deque<std::optional<std::complex<unsigned long int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned long int>>>>&,       char (*)(std::optional<std::deque<std::optional<std::complex<unsigned long int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned long int>>>> const&, char (*)(std::optional<std::deque<std::optional<std::complex<unsigned long int>>>> const&), char>,
    return_type_test_type<0, std::optional<std::deque<std::optional<std::complex<unsigned long int>>>>&&,      char (*)(std::optional<std::deque<std::optional<std::complex<unsigned long int>>>> const&), char>
>;

// -----------------------------------------------------------------------------
// Compile-time Type Assertion Framework
// -----------------------------------------------------------------------------

// Base template structure.
template <typename T>
struct check_return_type;

// Specialization to unpack our specific test tuple.
template <typename Container, typename Function, typename Expected, typename Result>
struct check_return_type<std::tuple<Container, Function, Expected, Result>>
{
    // If Expected and Result types do not match, compilation will fail here.
    static_assert(std::is_same_v<Expected, Result>, "Return type does not match the expected type.");
};

// Variadic template structure to evaluate all test cases in the outer tuple.
template <typename Tuple>
struct check_all_return_types;

// Specialization using a fold expression-like inheritance pattern to instantiate
// check_return_type for every element inside the std::tuple.
template <typename... Ts>
struct check_all_return_types<std::tuple<Ts...>> : check_return_type<Ts>...
{
};

// Explicit instantiation triggers the static_asserts at compile time.
template struct check_all_return_types<return_type_test_types>;


// -----------------------------------------------------------------------------
// Runtime Tests Framework
// -----------------------------------------------------------------------------

// Define a struct with a call operator instead of using inline lambdas, per preference.
struct ReturnMinusOne
{
    template <typename T>
    constexpr auto operator()(T&& /*unused*/) const
    {
        return -1;
    }
};

// Define a struct with a call operator constrained by standard concepts.
struct AddTwentySeven
{
    template <typename T>
    requires std::is_arithmetic_v<std::remove_cvref_t<T>>
    constexpr auto operator()(T&& v) const
    {
        return v + 27;
    }
};

void test_empty_1d_int_vector()
{
    const std::vector<int> input{};
    // Vector depth is 1, so unwrap_level is 1. Passed function is now the second argument.
    const auto result = TinyDIP::recursive_transform<1>(std::execution::par, ReturnMinusOne{}, input);

    // Output and validation
    if (result.empty())
    {
        std::cout << "test_empty_1d_int_vector: PASS\n";
    }
    else
    {
        std::cerr << "test_empty_1d_int_vector: FAIL\n";
    }
    
    // Hard check (abort execution if condition is false)
    assert(result.empty());
}

void test_simple_int()
{
    const auto input = 42;
    // Scalar depth is 0, so unwrap_level is 0. Passed function is now the second argument.
    const auto result = TinyDIP::recursive_transform<0>(std::execution::par, AddTwentySeven{}, input);

    // Output and validation
    if (result == 69)
    {
        std::cout << "test_simple_int: PASS. Expected output: 69, Actual output: " << result << "\n";
    }
    else
    {
        std::cerr << "test_simple_int: FAIL. Expected output: 69, Actual output: " << result << "\n";
    }
    
    // Hard check (abort execution if condition is false)
    assert(result == 69);
}

int main()
{
    TinyDIP::Timer timer1;
    std::cout << "Starting tests for TinyDIP::recursive_transform (Pure C++ STL)...\n\n";

    // Since 'check_all_return_types' was instantiated globally, if the program compiles, 
    // it inherently guarantees all type trait assertions passed.
    std::cout << "1. Compile-time return type tests: PASS\n\n";

    std::cout << "2. Runtime tests execution:\n";
    test_empty_1d_int_vector();
    test_simple_int();

    std::cout << "\nAll tests completed successfully.\n";

    return EXIT_SUCCESS;
}