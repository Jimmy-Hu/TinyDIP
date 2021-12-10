# TinyDIP

Tiny Digital Image Processing Library is a C++ library designed with C++20 syntax.

**Note: The implementation in this repository is under experimental purpose. The condition of requirment for development environment, library dependencies, even the used C++ language version may be updated.**

### Environment Requirment

Both `std::ranges` and [concepts](https://en.cppreference.com/w/cpp/language/constraints) ard widely used in TinyDIP library. C++ compiler support information could be checked with https://en.cppreference.com/w/cpp/compiler_support#cpp20. The suggested compiler (tested, minimum) version are listed as below.

- GCC: g++-11

- MSVC: MSVC 19.24

### Build Configuration & Process

#### Linux & MacOS

TinyDIP library could be built with [CMake](https://cmake.org/) as the following commands:

1. Create `./build` folder:
  
  ```shell
  mkdir build
  ```
  

#### Windows



### Code Review

- Two dimensional gaussian image generator in C++, Jun 25, 2021 @ GMT+8
  
  https://codereview.stackexchange.com/q/263422/231235

- Two dimensional bicubic interpolation implementation in C++, Jun 22, 2021 @ GMT+8
  
  https://codereview.stackexchange.com/q/263289/231235
  
- Some further details about "basic_functions" please check "docs/basic_functions/README.md"


