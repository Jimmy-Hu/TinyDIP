rm -rf build && mkdir build
cmake -B ./build -DCMAKE_TOOLCHAIN_FILE=../../vcpkg/scripts/buildsystems/vcpkg.cmake -S .
cmake --build ./build --parallel --verbose
./build/TinyDIP
