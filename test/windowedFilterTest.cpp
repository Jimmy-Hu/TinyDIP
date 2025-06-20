/* Developed by Jimmy Hu */

#include <chrono>
#include <execution>
#include <map>
#include <omp.h>
#include <sstream>
#include <tbb/global_control.h>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"

//  remove_extension Function Implementation
//  Copy from: https://stackoverflow.com/a/6417908/6667035
std::string remove_extension(const std::string& filename)
{
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    omp_set_num_threads(18); // Use 18 threads for all consecutive parallel regions
    
    tbb::global_control gc(
        tbb::global_control::max_allowed_parallelism, 4  // Limit to 4 threads
    );
    std::cout << "argc parameter: " << std::to_string(argc) << '\n';
    auto random_complex_image = TinyDIP::rand(10, 10, 1);
	std::cout << "Random complex image size: " << random_complex_image.getWidth() << "x" << random_complex_image.getHeight() << '\n';
	std::cout << "Random complex image dimensionality: " << random_complex_image.getDimensionality() << '\n';
	//std::cout << "Random complex image type: " << TinyDIP::getTypeName(random_complex_image.getType()) << '\n';
    random_complex_image.print();
    std::vector<std::size_t> window_sizes1{3, 3, 1};
    auto windowed_filter_output1 = TinyDIP::windowed_filter(
        std::execution::seq,
        random_complex_image,
        window_sizes1,
        [&](auto&& input_window) { return TinyDIP::mean(input_window); },
        TinyDIP::mirror
	);
    windowed_filter_output1.print();
    return EXIT_SUCCESS;
}