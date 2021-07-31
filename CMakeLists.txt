# Reference: https://cmake.org/cmake/help/latest/guide/tutorial/index.html
cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(TinyDIP VERSION 1.0)

# specify the C++ standard
# Reference: https://stackoverflow.com/q/60348039/6667035
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE debug)
# Reference: https://superuser.com/a/1424303/1147141
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors -std=c++20 -Xpreprocessor -fopenmp -O3")
## Reference: https://stackoverflow.com/a/45934279/6667035
set(CMAKE_C_COMPILER "/usr/local/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-11")

# add the executable
add_executable(TinyDIP main.cpp image_io.cpp)

target_include_directories(TinyDIP PRIVATE ${LLVM_INCLUDE_DIRS})

# Reference: https://stackoverflow.com/a/39600062/6667035
target_link_libraries(TinyDIP -lm)
#target_link_libraries(TinyDIP -lomp)
target_link_libraries(TinyDIP -lpthread)
#target_link_libraries(TinyDIP -L/usr/local/Cellar/llvm/10.0.0_3/lib/)

# Add boost library
# Reference: https://github.com/Microsoft/vcpkg/issues/4188
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED OFF)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(BOOST 76.0 REQUIRED COMPONENTS serialization test)
find_path(BOOST_INCLUDE_DIR boost)
include_directories(${BOOST_INCLUDE_DIR})

target_link_libraries(TinyDIP ${BOOST_LIBRARIES})
