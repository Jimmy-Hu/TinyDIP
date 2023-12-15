/* Developed by Jimmy Hu */

#include <execution>
#include <stdlib.h>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

//  Copy from https://stackoverflow.com/a/37264642/6667035
#ifndef NDEBUG
#   define M_Assert(Expr, Msg) \
    __M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)
#else
#   define M_Assert(Expr, Msg) ;
#endif

void __M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg)
{
    if (!expr)
    {
        std::cerr << "Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

void recursive_transform_reduce_tests()
{
    auto test_vectors_1 = TinyDIP::n_dim_container_generator<1, int, std::vector>(1, 3);
    //  basic usage case
    M_Assert(TinyDIP::recursive_transform_reduce<1>(test_vectors_1, 0) == 3, "Basic usage case failed");

    //  basic usage case with execution policy
    M_Assert(TinyDIP::recursive_transform_reduce<1>(std::execution::par, test_vectors_1, 0) == 3,
        "Basic usage case with execution policy failed");

    //  test case with unary operation
    M_Assert(TinyDIP::recursive_transform_reduce<1>(
        test_vectors_1,
        0,
        [&](auto&& element) { return element + 1; }) == 6,
        "Test case with unary operation failed");

    //  test case with unary operation, execution policy
    M_Assert(TinyDIP::recursive_transform_reduce<1>(
        std::execution::par,
        test_vectors_1,
        0,
        [&](auto&& element) { return element + 1; }) == 6,
        "Test case with unary operation, execution policy failed");

    //  test case with unary operation and binary operation
    M_Assert(TinyDIP::recursive_transform_reduce<1>(
        test_vectors_1,
        1,
        [&](auto&& element) { return element + 1; },
        [&](auto&& element1, auto&& element2) { return element1 * element2; }) == 8,
        "Test case with unary operation and binary operation failed");

    //  test case with unary operation, binary operation and execution policy
    M_Assert(TinyDIP::recursive_transform_reduce<1>(
        std::execution::par,
        test_vectors_1,
        1,
        [&](auto&& element) { return element + 1; },
        [&](auto&& element1, auto&& element2) { return element1 * element2; }) == 8,
        "Test case with unary operation, binary operation and execution policy failed");

    auto test_string_vector_1 = TinyDIP::n_dim_container_generator<1, std::string, std::vector>("1", 3);
    //  test case with std::string
    M_Assert(TinyDIP::recursive_transform_reduce<1>(test_string_vector_1, std::string("")) == "111",
        "Test case with std::string failed");

    //  test case with std::string, execution policy
    M_Assert(recursive_transform_reduce<1>(
        std::execution::par,
        test_string_vector_1, std::string("")) == "111",
        "Test case with std::string, execution policy failed");

    //  test case with std::string, unary operation
    M_Assert(recursive_transform_reduce<1>(
        test_string_vector_1,
        std::string(""),
        [&](auto&& element) { return element + "2";}) == "121212",
        "Test case with std::string, unary operation failed");

    //  test case with std::string, unary operation, execution policy
    M_Assert(recursive_transform_reduce<1>(
        std::execution::par,
        test_string_vector_1,
        std::string(""),
        [&](auto&& element) { return element + "2";}) == "121212",
        "Test case with std::string, unary operation, execution policy failed");

    //  test case with nested std::vector
    std::vector<decltype(test_vectors_1)> test_vectors_2 = {test_vectors_1, test_vectors_1};
    M_Assert(recursive_transform_reduce<2>(test_vectors_2, 1) == 7,
        "Test case with nested std::vector failed");

    //  test case with nested std::vector, execution policy
    M_Assert(recursive_transform_reduce<2>(std::execution::par, test_vectors_2, 1) == 7,
        "Test case with nested std::vector, execution policy failed");

    //  test case with nested std::array
    std::array<int, 3> test_array_1 = {1, 1, 1};
    std::array<decltype(test_array_1), 2> test_array_2 = {test_array_1, test_array_1};
    M_Assert(recursive_transform_reduce<2>(test_array_2, 1) == 7,
        "Test case with nested std::vector failed");

    //  test case with nested std::array, execution policy
    M_Assert(recursive_transform_reduce<2>(std::execution::par, test_array_2, 1) == 7,
        "Test case with nested std::vector, execution policy failed");

    //  test case with nested std::deque
    auto test_deque_1 = n_dim_container_generator<1, int, std::deque>(1, 3);
    std::deque<decltype(test_deque_1)> test_deque_2 = {test_deque_1, test_deque_1};
    M_Assert(recursive_transform_reduce<2>(test_deque_2, 1) == 7,
        "Test case with nested std::deque failed");
    
    //  test case with nested std::deque, execution policy
    M_Assert(recursive_transform_reduce<2>(std::execution::par, test_deque_2, 1) == 7,
        "Test case with nested std::deque, execution policy failed");

    //  test case with nested std::list
    auto test_list_1 = n_dim_container_generator<1, int, std::list>(1, 3);
    std::list<decltype(test_list_1)> test_list_2 = {test_list_1, test_list_1};
    M_Assert(recursive_transform_reduce<2>(test_list_2, 1) == 7,
        "Test case with nested std::list failed");

    //  test case with nested std::list, execution policy
    M_Assert(recursive_transform_reduce<2>(std::execution::par, test_list_2, 1) == 7,
        "Test case with nested std::list failed");

    std::cout << "All tests passed!\n";

    return;
}

int main()
{
    auto start = std::chrono::system_clock::now();
    recursive_transform_reduce_tests();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}