/* Developed by Jimmy Hu */

#include <cassert>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
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

void recursive_remove_copy_if_tests()
{
    //  std::vector<int> test case
    std::vector<int> test_vector_1 = {
        1, 2, 3, 4, 5, 6
    };
    std::vector<int> expected_result_1 = {
        1, 3, 5
    };
    M_Assert(
        TinyDIP::recursive_remove_copy_if<1>(test_vector_1, [](auto&& x) { return (x % 2) == 0; }) ==
        expected_result_1,
        "std::vector<int> test case failed");
    
    //  std::vector<std::vector<int>> test case
    std::vector<decltype(test_vector_1)> test_vector_2 = {
        test_vector_1, test_vector_1, test_vector_1
    };
    std::vector<std::vector<int>> expected_result_2 = {
        expected_result_1, expected_result_1, expected_result_1
    };
    M_Assert(
        TinyDIP::recursive_remove_copy_if<2>(test_vector_2, [](auto&& x) { return (x % 2) == 0; }) ==
        expected_result_2,
        "std::vector<std::vector<int>> test case failed");

    //  std::vector<std::string> test case
    std::vector<std::string> test_vector_3 = {
        "1", "2", "3", "4", "5", "6"
    };
    std::vector<std::string> expected_result_3 = {
        "2", "3", "4", "5", "6"
    };
    M_Assert(
        TinyDIP::recursive_remove_copy_if<1>(test_vector_3, [](auto&& x) { return (x == "1"); }) ==
        expected_result_3,
        "std::vector<std::string> test case failed");

    //  std::vector<std::vector<std::string>> test case
    std::vector<std::vector<std::string>> test_vector_4 = {
        test_vector_3, test_vector_3, test_vector_3
    };
    std::vector<std::vector<std::string>> expected_result_4 = {
        expected_result_3, expected_result_3, expected_result_3
    };
    M_Assert(
        TinyDIP::recursive_remove_copy_if<2>(test_vector_4, [](auto&& x) { return (x == "1"); }) ==
        expected_result_4,
        "std::vector<std::vector<std::string>> test case failed");

    //  std::deque<int> test case
    std::deque<int> test_deque_1;
    test_deque_1.push_back(1);
    test_deque_1.push_back(2);
    test_deque_1.push_back(3);
    test_deque_1.push_back(4);
    test_deque_1.push_back(5);
    test_deque_1.push_back(6);
    std::deque<int> expected_result_5;
    expected_result_5.push_back(2);
    expected_result_5.push_back(3);
    expected_result_5.push_back(4);
    expected_result_5.push_back(5);
    expected_result_5.push_back(6);
    M_Assert(
        TinyDIP::recursive_remove_copy_if<1>(test_deque_1, [](auto&& x) { return (x == 1); }) ==
        expected_result_5,
        "std::deque<int> test case failed"
    );

    //  std::deque<std::deque<int>> test case
    std::deque<decltype(test_deque_1)> test_deque_2;
    test_deque_2.push_back(test_deque_1);
    test_deque_2.push_back(test_deque_1);
    test_deque_2.push_back(test_deque_1);
    std::deque<decltype(expected_result_5)> expected_result_6;
    expected_result_6.push_back(expected_result_5);
    expected_result_6.push_back(expected_result_5);
    expected_result_6.push_back(expected_result_5);
    M_Assert(
        TinyDIP::recursive_remove_copy_if<2>(test_deque_2, [](auto&& x) { return (x == 1); }) ==
        expected_result_6,
        "std::deque<std::deque<int>> test case failed"
    );

    //  std::list<int> test case
    std::list<int> test_list_1 = { 1, 2, 3, 4, 5, 6 };
    std::list<int> expected_result_7 = {1, 3, 5};
    M_Assert(
        TinyDIP::recursive_remove_copy_if<1>(test_list_1, [](int x) { return (x % 2) == 0; }) ==
        expected_result_7,
        "std::list<int> test case failed"
    );

}

int main()
{
    auto start = std::chrono::system_clock::now();
    recursive_remove_copy_if_tests();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}