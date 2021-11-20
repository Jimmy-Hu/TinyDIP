# Reference: https://cmake.org/cmake/help/latest/guide/tutorial/index.html
cmake_minimum_required(VERSION 3.10)

# set the project name and version
# Reference: https://stackoverflow.com/a/41651622/6667035
project(TinyDIP VERSION 1.0 LANGUAGES C CXX)

# print compile/link commands
# Reference: https://stackoverflow.com/a/3379246/6667035
set(CMAKE_VERBOSE_MAKEFILE on)

# specify the C++ standard
# Reference: https://stackoverflow.com/q/60348039/6667035
# Reference: https://stackoverflow.com/a/53657598/6667035
IF (CMAKE_VERSION VERSION_LESS "3.8")
    SET(CMAKE_CXX_STANDARD 14)
ELSEIF (CMAKE_VERSION VERSION_LESS "3.11")
    SET(CMAKE_CXX_STANDARD 17)
ELSE()
    SET(CMAKE_CXX_STANDARD 20)
ENDIF()
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE debug)

## Reference: https://stackoverflow.com/a/45934279/6667035
if(APPLE)           # Reference: https://stackoverflow.com/a/40152725/6667035
    set(CMAKE_C_COMPILER "/usr/local/bin/gcc")
    set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-11")
endif()

# Using different command for compiling
# Reference: https://stackoverflow.com/a/10055571/6667035
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  # Reference: https://superuser.com/a/1424303/1147141
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors -std=c++20 -Xpreprocessor -fopenmp -O3")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /Ox /Ob2 /Oi /Ot /GT /std:c++latest /experimental:module /Zm2000 /Zc:twoPhase- /D_HAS_STD_BYTE=0 /D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING /D_HAS_STD_BOOLEAN=0 /bigobj ")
endif()

# OpenMP part
# Reference: https://stackoverflow.com/a/12404666/6667035
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# add the executable
add_executable(TinyDIP main.cpp image_io.cpp)
add_executable(bicubicInterpolationTest test/bicubicInterpolationTest.cpp image_io.cpp)
add_executable(subimageTest test/subimageTest.cpp image_io.cpp)
add_executable(subimage2Test test/subimage2Test.cpp image_io.cpp)
add_executable(rgb2hsvTest test/rgb2hsvTest.cpp image_io.cpp)
add_executable(increaseIntensityTest test/increaseIntensityTest.cpp image_io.cpp)
add_executable(concatHorizontalTest test/concatHorizontalTest.cpp image_io.cpp)
add_executable(concatVerticalTest test/concatVerticalTest.cpp image_io.cpp)
add_executable(concatTest test/concatTest.cpp image_io.cpp)
add_executable(dct3DetailTest test/dct3DetailTest.cpp image_io.cpp)
add_executable(dct3Test test/dct3Test.cpp image_io.cpp)
add_executable(idct3DetailTest test/idct3DetailTest.cpp image_io.cpp)
add_executable(isIntegerTest test/isIntegerTest.cpp image_io.cpp)

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
find_package(BOOST 76.0 REQUIRED COMPONENTS iterator serialization test)
find_path(BOOST_INCLUDE_DIR boost)
include_directories(${BOOST_INCLUDE_DIR})

target_link_libraries(TinyDIP Boost::boost ${BOOST_LIBRARIES})

# Reference: https://stackoverflow.com/a/31390758/6667035
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

cmake_language(CALL print_all_variables)

