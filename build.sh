#!/bin/bash
clear
# Remove build directory (Avoid sudo if possible, it messes up permissions)
rm -rf build
mkdir build

# Configure CMake
# Removed explicit -DCMAKE_CXX_COMPILER=/usr/bin/g++ on macOS 
# because /usr/bin/g++ is usually Apple Clang, not GCC.
# Allowing CMake to auto-detect usually works best, especially with the fixed CMakeLists.txt.

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS. Using default compiler (likely AppleClang) with libomp detection."
    cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -S . -B ./build
else
    # On Linux, forcing gcc/g++ is generally fine if you want to bypass system clang
    cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -S . -B ./build
fi

# Build
cmake --build ./build --parallel --verbose

# Run executables
./build/TinyDIP
./build/recursiveTransformTest
