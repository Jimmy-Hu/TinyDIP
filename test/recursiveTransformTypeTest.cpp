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
template <typename Container, typename Function, typename Expected>
using return_type_test_type = std::tuple<
    Container,
    Function,
    Expected,
    decltype(TinyDIP::recursive_transform(std::execution::par, std::declval<Container>(), std::declval<Function>()))
>;

// The collection of types to test.
using return_type_test_types = std::tuple<
    //                    Container   Function                Result

    // Non-range.
    //  template
    //  Replace "\n//T1 replace finished.\r\n\r\n" into "\n"
    /*
    return_type_test_type<T1,        T2 (*)(T1),        T2>,
    return_type_test_type<T1,        T2 (*)(T1 const&), T2>,
    return_type_test_type<T1 const,  T2 (*)(T1 const&), T2>,
    return_type_test_type<T1&,       T2 (*)(T1 const&), T2>,
    return_type_test_type<T1 const&, T2 (*)(T1 const&), T2>,
    return_type_test_type<T1&&,      T2 (*)(T1 const&), T2>,
    */
    return_type_test_type<char,        char (*)(char),        char>,
    return_type_test_type<char,        char (*)(char const&), char>,
    return_type_test_type<char const,  char (*)(char const&), char>,
    return_type_test_type<char&,       char (*)(char const&), char>,
    return_type_test_type<char const&, char (*)(char const&), char>,
    return_type_test_type<char&&,      char (*)(char const&), char>,

    return_type_test_type<int,        char (*)(int),        char>,
    return_type_test_type<int,        char (*)(int const&), char>,
    return_type_test_type<int const,  char (*)(int const&), char>,
    return_type_test_type<int&,       char (*)(int const&), char>,
    return_type_test_type<int const&, char (*)(int const&), char>,
    return_type_test_type<int&&,      char (*)(int const&), char>,

    return_type_test_type<short,        char (*)(short),        char>,
    return_type_test_type<short,        char (*)(short const&), char>,
    return_type_test_type<short const,  char (*)(short const&), char>,
    return_type_test_type<short&,       char (*)(short const&), char>,
    return_type_test_type<short const&, char (*)(short const&), char>,
    return_type_test_type<short&&,      char (*)(short const&), char>,

    return_type_test_type<long,        char (*)(long),        char>,
    return_type_test_type<long,        char (*)(long const&), char>,
    return_type_test_type<long const,  char (*)(long const&), char>,
    return_type_test_type<long&,       char (*)(long const&), char>,
    return_type_test_type<long const&, char (*)(long const&), char>,
    return_type_test_type<long&&,      char (*)(long const&), char>,

    return_type_test_type<long long int,        char (*)(long long int),        char>,
    return_type_test_type<long long int,        char (*)(long long int const&), char>,
    return_type_test_type<long long int const,  char (*)(long long int const&), char>,
    return_type_test_type<long long int&,       char (*)(long long int const&), char>,
    return_type_test_type<long long int const&, char (*)(long long int const&), char>,
    return_type_test_type<long long int&&,      char (*)(long long int const&), char>
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
    const auto result = TinyDIP::recursive_transform(std::execution::par, input, ReturnMinusOne{});

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
    const auto result = TinyDIP::recursive_transform(std::execution::par, input, AddTwentySeven{});

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