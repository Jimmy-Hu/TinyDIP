//  recursiveFoldRightAllTest.cpp

#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

void recursiveFoldRightAllTest();

int main()
{
    auto start = std::chrono::system_clock::now();
	recursiveFoldRightAllTest();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
	return 0;
}

void recursiveFoldRightAllTest()
{
    auto test_vectors = TinyDIP::n_dim_container_generator<4>(1, 3);

    std::cout << "Play with test_vectors:\n\n";
    
    std::cout << "recursive_fold_right_all function test with vectors / std::plus<>(): \n";
    auto recursive_fold_right_all_result1 = TinyDIP::recursive_fold_right_all(test_vectors, static_cast<int>(1), std::plus<>());
    
    return;
}

