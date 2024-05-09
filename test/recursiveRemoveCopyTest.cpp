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

//  recursive_remove_copy_execution_policy_tests function implementation
void recursive_remove_copy_execution_policy_tests()
{
    //  std::vector<int> test case
    std::vector<int> test_vector_1 = {
        1, 2, 3, 4, 5, 6
    };
    std::vector<int> expected_result_1 = {
        2, 3, 4, 5, 6
    };
    M_Assert(
        TinyDIP::recursive_remove_copy<1>(std::execution::par, test_vector_1, 1) ==
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
        TinyDIP::recursive_remove_copy<2>(std::execution::par, test_vector_2, 1) ==
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
        TinyDIP::recursive_remove_copy<1>(std::execution::par, test_vector_3, "1") ==
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
        TinyDIP::recursive_remove_copy<2>(std::execution::par, test_vector_4, "1") ==
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
    
    std::cout << "All tests passed!\n";
}

int main()
{
    auto start = std::chrono::system_clock::now();
    recursive_remove_copy_execution_policy_tests();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}