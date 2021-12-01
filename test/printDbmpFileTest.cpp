#include <cassert>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

static void print_dbmp(std::string path)
{
    auto dbmp_input = TinyDIP::double_image::read(path.c_str(), false);
    dbmp_input.print();
    return;
}

static void print_dbmp(std::string path, std::size_t startx, std::size_t endx, std::size_t starty, std::size_t endy)
{
    auto dbmp_input = TinyDIP::double_image::read(path.c_str(), false);
    TinyDIP::subimage2(dbmp_input, startx, endx, starty, endy).print();
    return;
}

static std::size_t string2size(std::string input)
{
    std::stringstream ss(input);
    std::size_t output;
    ss >> output;
    return output;
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
	else if(argc == 6)
	{
		auto arg1 = std::string(argv[1]);
        auto arg2 = string2size(std::string(argv[2]));
        auto arg3 = string2size(std::string(argv[3]));
        auto arg4 = string2size(std::string(argv[4]));
        auto arg5 = string2size(std::string(argv[5]));
		print_dbmp(arg1, arg2, arg3, arg4, arg5);
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