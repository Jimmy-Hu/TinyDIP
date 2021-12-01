#include <cassert>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

constexpr static void print_dbmp(std::string path)
{
    auto dbmp_input = TinyDIP::double_image::read(path.c_str(), false);
    dbmp_input.print();
    return;
} 

int main(int argc, char* argv[])
{
	auto start = std::chrono::system_clock::now();
	std::cout << std::to_string(argc) << '\n';
	if (argc == 2)
	{
		auto arg1 = std::string(argv[1]);
		print_dbmp(arg1);
	}
	else
	{
		
	}
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Execution finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
	return 0;
}