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

# 2. Define the exact regex filter to EXCLUDE system-locked libraries (Blacklist approach)
# We must NOT copy kernel-bound GLIBC components or hardware-specific graphics drivers.
# Everything else (OpenCV, FFmpeg, VTK, Qt5, Custom GCC) will be bundled automatically!
BLACKLIST_REGEX="(libc\.so|libm\.so|libdl\.so|libpthread\.so|librt\.so|libresolv\.so|libnsl\.so|libutil\.so|libcrypt\.so|libGL\.so|libGLX\.so|libEGL\.so|libGLdispatch\.so|libdrm\.so|libX11|libxcb|libXext|libXrender|libXfixes|libXau|libXdmcp|libudev\.so|ld-linux|libvulkan|libgbm|libwayland)"

# 3. Harvest from the main executable
# awk 'NF == 4 {print $3}' safely extracts the absolute paths, ignoring virtual VDSO.
# grep -v -E inverts the match to exclude the blacklist.
if [ -f "./build/TinyDIP" ]; then
    echo "Harvesting dependencies for TinyDIP..."
    ldd ./build/TinyDIP | awk 'NF == 4 {print $3}' | grep -v -E "$BLACKLIST_REGEX" | xargs -I '{}' cp -vn '{}' ./build/lib/
fi

# 4. Harvest from any compiled plugins
if [ -d "./build/plugins" ]; then
    echo "Harvesting dependencies for dynamic plugins..."
    for plugin in ./build/plugins/*.so; do
        if [ -f "$plugin" ]; then
            ldd "$plugin" | awk 'NF == 4 {print $3}' | grep -v -E "$BLACKLIST_REGEX" | xargs -I '{}' cp -vn '{}' ./build/lib/
        fi
    done
fi

# 5. Generate a portable launch wrapper script
# This solves the RUNPATH scoping issue by forcing all deep dependencies to check the lib folder first!
echo "Generating launch.sh wrapper..."
cat << 'EOF' > ./build/launch.sh
#!/bin/sh
# Industry standard wrapper to enforce local library prioritization
# Utilizing $0 natively ensures POSIX compliance so it resolves perfectly under 'sh', 'bash', or 'dash'
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"
exec "$SCRIPT_DIR/TinyDIP" "$@"
EOF
chmod +x ./build/launch.sh

echo "Deployment package successfully assembled in ./build/lib/"
# ------------------------------------------------------------------------------------

# Run executables using the new isolated launch script
./build/launch.sh
./build/recursiveTransformTest