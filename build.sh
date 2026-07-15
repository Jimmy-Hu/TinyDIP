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

# ------------------------------------------------------------------------------------
#  Automated Dependency Harvesting for Portable Deployment
# ------------------------------------------------------------------------------------
echo "Packaging shared libraries for deployment..."

# 1. Create the deployment library directory
mkdir -p ./build/lib

# 2. Define the exact regex filter for safe, portable libraries
# We explicitly want OpenCV, TBB, GCC C++ runtimes, HDF5, Qt5, VTK, and other image codecs.
SAFE_LIBS_REGEX="(libopencv|libtbb|libstdc\+\+|libgomp|libgcc_s|libhdf5|libQt5|libvtk|libopenblas|libgfortran|libpng|libjpeg|libtiff|libwebp)"

# 3. Harvest from the main executable
if [ -f "./build/TinyDIP" ]; then
    echo "Harvesting dependencies for TinyDIP..."
    ldd ./build/TinyDIP | awk '{print $3}' | grep -E $SAFE_LIBS_REGEX | xargs -I '{}' cp -vn '{}' ./build/lib/
fi

# 4. Harvest from any compiled plugins
if [ -d "./build/plugins" ]; then
    echo "Harvesting dependencies for dynamic plugins..."
    for plugin in ./build/plugins/*.so; do
        if [ -f "$plugin" ]; then
            ldd "$plugin" | awk '{print $3}' | grep -E $SAFE_LIBS_REGEX | xargs -I '{}' cp -vn '{}' ./build/lib/
        fi
    done
fi

echo "Deployment package successfully assembled in ./build/lib/"
# ------------------------------------------------------------------------------------

# Run executables
./build/TinyDIP
./build/recursiveTransformTest
