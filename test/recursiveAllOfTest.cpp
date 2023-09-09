/* Developed by Jimmy Hu */

#include <execution>
#include <stdlib.h>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

template<std::size_t dim>
void recursive_all_of_tests();

int main()
{
    auto start = std::chrono::system_clock::now();
	recursive_all_of_tests<4>();
    recursive_all_of_tests<5>();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return EXIT_SUCCESS;
}

//  recursive_all_of_tests template function implementation
template<std::size_t dim>
void recursive_all_of_tests()
{
    if constexpr(dim == 4)
    {
        auto test_vectors_1 = TinyDIP::n_dim_container_generator<dim, int, std::vector>(1, 3);
        test_vectors_1[0][0][0][0] = 2;
        assert(TinyDIP::recursive_all_of<dim>(test_vectors_1, [](auto&& i) { return i % 2 == 0; }) == false);

        auto test_vectors_2 = TinyDIP::n_dim_container_generator<dim, int, std::vector>(2, 3);
        test_vectors_2[0][0][0][0] = 4;
        assert(TinyDIP::recursive_all_of<dim>(test_vectors_2, [](auto&& i) { return i % 2 == 0; }));
    }
    
    //  Tests with std::string
    auto test_vector_string = TinyDIP::n_dim_container_generator<dim, std::string, std::vector>("1", 3);
    assert(TinyDIP::recursive_all_of<dim>(test_vector_string, [](auto&& i) { return i == "1"; }));
    assert(TinyDIP::recursive_all_of<dim>(test_vector_string, [](auto&& i) { return i == "2"; }) == false);

    //  Tests with std::string, projection
    assert(TinyDIP::recursive_all_of<dim>(
        test_vector_string,
        [](auto&& i) { return i == "1"; },
        [](auto&& element) {return std::to_string(stoi(element) + 1); }) == false);
    assert(TinyDIP::recursive_all_of<dim>(
        test_vector_string,
        [](auto&& i) { return i == "2"; },
        [](auto&& element) {return std::to_string(stoi(element) + 1); }));
    
    //  Tests with std::array of std::string
    std::array<std::string, 3> word_array1 = {"foo", "foo", "foo"};
    assert(TinyDIP::recursive_all_of<1>(word_array1, [](auto&& i) { return i == "foo"; }));
    assert(TinyDIP::recursive_all_of<1>(word_array1, [](auto&& i) { return i == "bar"; }) == false);
    assert(TinyDIP::recursive_all_of<2>(word_array1, [](auto&& i) { return i == 'a'; }) == false);

    //  Tests with std::deque of std::string
    std::deque<std::string> word_deque1 = {"foo", "foo", "foo", "foo"};
    assert(TinyDIP::recursive_all_of<1>(word_deque1, [](auto&& i) { return i == "foo"; }));
    assert(TinyDIP::recursive_all_of<1>(word_deque1, [](auto&& i) { return i == "bar"; }) == false);

    std::vector<std::wstring> wstring_vector1{};
    for(int i = 0; i < 4; ++i)
    {
        wstring_vector1.push_back(std::to_wstring(1));
    }
    assert(TinyDIP::recursive_all_of<1>(wstring_vector1, [](auto&& i) { return i == std::to_wstring(1); }));
    assert(TinyDIP::recursive_all_of<1>(wstring_vector1, [](auto&& i) { return i == std::to_wstring(2); }) == false);

    std::vector<std::u8string> u8string_vector1{};
    for(int i = 0; i < 4; ++i)
    {
        u8string_vector1.push_back(u8"\u20AC2.00");
    }
    assert(TinyDIP::recursive_all_of<1>(u8string_vector1, [](auto&& i) { return i == u8"\u20AC2.00"; }));
    assert(TinyDIP::recursive_all_of<1>(u8string_vector1, [](auto&& i) { return i == u8"\u20AC1.00"; }) == false);

    std::pmr::string pmr_string1 = "123";
    std::vector<std::pmr::string> pmr_string_vector1 = {pmr_string1, pmr_string1, pmr_string1};
    assert(TinyDIP::recursive_all_of<1>(pmr_string_vector1, [](auto&& i) { return i == "123"; }));
    assert(TinyDIP::recursive_all_of<1>(pmr_string_vector1, [](auto&& i) { return i == "456"; }) == false);
    std::cout << "All tests passed!\n";

    return;
}