# TinyDIP

Tiny Digital Image Processing Library is a C++ library designed with C++20 syntax.

**Note: The implementation in this repository is under experimental purpose. The condition of requirment for development environment, library dependencies, even the used C++ language version may be updated.**

### Environment Requirment

Both `std::ranges` and [concepts](https://en.cppreference.com/w/cpp/language/constraints) are widely used in TinyDIP library. C++ compiler support information could be checked with https://en.cppreference.com/w/cpp/compiler_support#cpp20. The suggested compiler (tested, minimum) version are listed as below.

- GCC: g++-11

- MSVC: MSVC 19.24 (Visual Studio 2019 Update 4, Visual Studio 2019 version 16.4)

### Build Configuration & Process

#### Linux & MacOS

TinyDIP library could be built with [CMake](https://cmake.org/) as the following commands:

1. Create `./build` folder:
  
    ```shell
    mkdir build
    ```

2. Config project

    ```shell
    cmake -B -S .
    cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -S . -B ./build    #  specify CMAKE_C_COMPILER and CMAKE_CXX_COMPILER flags
    ```
    
    - integrated with `vcpkg` toolchain

    ```shell
    cmake -B ./build -DCMAKE_TOOLCHAIN_FILE=../../vcpkg/scripts/buildsystems/vcpkg.cmake -S .
    cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -S . -B ./build    #  specify CMAKE_C_COMPILER and CMAKE_CXX_COMPILER flags
    ```

3. Build project
    
    ```shell
    cmake --build ./build
    ```

    If the parallel build process is prefered, `--parallel` flag can be added. However, if you want to build in parallel, ensure the memory is large enough. On the other hand, if the verbose information is needed, `--verbose` flag can be added.

    ```shell
    cmake --build ./build --parallel --verbose
    ```

4. Run the built execution file

    After the building process as above, the built execution files can be checked under `./build/` folder. The built execution file `TinyDIP` can be run with the following command:

    ```shell
    ./build/TinyDIP
    ```

#### Windows

- In Visual Studio IDE, [CMake projects can be opened in Visual Studio](https://docs.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=msvc-170)

### Code Review

- Two dimensional gaussian image generator in C++, Jun. 25, 2021 @ GMT+8
  
  https://codereview.stackexchange.com/q/263422/231235

- Two dimensional bicubic interpolation implementation in C++, Jun. 22, 2021 @ GMT+8
  
  https://codereview.stackexchange.com/q/263289/231235
  
- Image Rotation and Transpose Functions Implementation in C++, Mar. 26, 2024 @ GMT+8
  
  https://codereview.stackexchange.com/q/291249/231235
  
- Some further details about "basic_functions" please check "docs/basic_functions/README.md"


