clear && rm -rf build && mkdir build
cmake -B ./build -DCMAKE_TOOLCHAIN_FILE=../../vcpkg/scripts/buildsystems/vcpkg.cmake -S .
cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -S . -B ./build
cmake --build ./build --parallel --verbose
./build/TinyDIP
