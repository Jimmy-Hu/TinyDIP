#!/bin/bash
clear
# Remove build directory (Avoid sudo if possible, it messes up permissions)
rm -rf build
mkdir build

# Configure CMake
# Added -DCMAKE_CXX_FLAGS="-fdiagnostics-color=always" to force compiler colorful output.
# Added -DCMAKE_COLOR_DIAGNOSTICS=ON to force CMake diagnostic colorful output.
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS. Using default compiler with forced color output."
    cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_COLOR_DIAGNOSTICS=ON -DCMAKE_CXX_FLAGS="-fdiagnostics-color=always" -S . -B ./build
else
    echo "Detected Linux. Using GCC/G++ with forced color output."
    cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_COLOR_DIAGNOSTICS=ON -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CXX_FLAGS="-fdiagnostics-color=always" -S . -B ./build
fi

# Build
# Redirect standard error (2) to standard output (1), and use 'tee' to both write to a file and display on the console.
# We use 'env CLICOLOR_FORCE=1' to strictly force CMake and underlying build systems (like Make/Ninja) to emit ANSI colors even when piped.
echo "Building project... (Output and warnings are being recorded to build_log.txt)"
env CLICOLOR_FORCE=1 cmake --build ./build --parallel --verbose -j10 2>&1 | tee build_log.txt

# Run executables
./build/TinyDIP
./build/recursiveTransformTest
