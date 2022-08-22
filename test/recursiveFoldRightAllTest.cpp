//  recursiveFoldRightAllTest.cpp

#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void recursiveFoldRightAllTest();

template<std::size_t dim>
void recursiveFoldRightAllTestWithVector(std::size_t times);

template<std::size_t dim, std::size_t times>
void recursiveFoldRightAllTestWithArray();

void recursiveFoldRightAllTestWithDeque();

int main()
{
    auto start = std::chrono::system_clock::now();
	recursiveFoldRightAllTest();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return 0;
}

void recursiveFoldRightAllTest()
{
    recursiveFoldRightAllTestWithVector<4>(1);
    recursiveFoldRightAllTestWithVector<5>(1);
    recursiveFoldRightAllTestWithVector<6>(1);

    recursiveFoldRightAllTestWithArray<4, 1>();
    recursiveFoldRightAllTestWithArray<5, 1>();
    recursiveFoldRightAllTestWithArray<6, 1>();
    
    auto v = {1, 2, 3, 4, 5, 6, 7, 8};
    std::string initial_string = "A";
    // Use a program defined function object (lambda-expression):
    std::string recursive_fold_right_all_result4 = TinyDIP::recursive_fold_right_all
    (
        v, initial_string, [](int x, std::string s) { return s + ':' + std::to_string(x); }
    );
    std::cout << "recursive_fold_right_all_result4: " << recursive_fold_right_all_result4 << '\n';

    return;
}

template<std::size_t dim>
void recursiveFoldRightAllTestWithVector(std::size_t times)
{
    auto test_vectors = TinyDIP::n_dim_container_generator<dim>(1, times);

    std::cout << "Play with test_vectors:\n\n";
    
    std::cout << "recursive_fold_right_all function test with vectors / std::plus<>(): \n";
    auto recursive_fold_right_all_result1 = TinyDIP::recursive_fold_right_all(test_vectors, static_cast<int>(1), std::plus<>());
    std::cout << recursive_fold_right_all_result1 << "\n\n";

    std::cout << "recursive_fold_right_all function test with vectors / std::minus<>(): \n";
    auto recursive_fold_right_all_result2 = TinyDIP::recursive_fold_right_all(test_vectors, static_cast<int>(2), std::minus<>());
    std::cout << recursive_fold_right_all_result2 << "\n\n";

    std::cout << "recursive_fold_right_all function test with vectors / std::multiplies<>(): \n";
    auto recursive_fold_right_all_result3 = TinyDIP::recursive_fold_right_all(test_vectors, static_cast<int>(2), std::multiplies<>());
    std::cout << recursive_fold_right_all_result3 << "\n\n";
}

template<std::size_t dim, std::size_t times>
void recursiveFoldRightAllTestWithArray()
{
    auto test_array1 = TinyDIP::n_dim_array_generator<dim, times>(1);

    std::cout << "Play with test_array1:\n\n";
    std::cout << "recursive_fold_right_all function test with array / std::plus<>(): \n";
    std::cout << TinyDIP::recursive_fold_right_all(test_array1, static_cast<int>(1), std::plus<>()) << "\n\n";

    std::cout << "recursive_fold_right_all function test with array / std::minus<>(): \n";
    std::cout << TinyDIP::recursive_fold_right_all(test_array1, static_cast<int>(2), std::minus<>()) << "\n\n";

    std::cout << "recursive_fold_right_all function test with array / std::multiplies<>(): \n";
    std::cout <<  TinyDIP::recursive_fold_right_all(test_array1, static_cast<int>(2), std::multiplies<>()) << "\n\n";
}
