# Reference: https://cmake.org/cmake/help/latest/guide/tutorial/index.html
cmake_minimum_required(VERSION 3.13)        # For lgtm.com service

# set the project name and version
# Reference: https://stackoverflow.com/a/41651622/6667035
# Start with C and CXX, CUDA will be enabled conditionally
project(TinyDIP VERSION 1.0 LANGUAGES C CXX)

# print compile/link commands
# Reference: https://stackoverflow.com/a/3379246/6667035
set(CMAKE_VERBOSE_MAKEFILE on)

# --- CUDA Setup ---
# Option to enable/disable CUDA, default is ON
OPTION(ENABLE_CUDA "Enable CUDA support" ON)

# Disable CUDA on Apple platforms as it's not officially supported
if(APPLE)
    message(STATUS "CUDA is not supported on macOS. Disabling CUDA build.")
    set(ENABLE_CUDA OFF)
endif()

# Set C++ standard and build type early
IF (CMAKE_VERSION VERSION_LESS "3.8")
    SET(CMAKE_CXX_STANDARD 14)
ELSEIF (CMAKE_VERSION VERSION_LESS "3.11")
    SET(CMAKE_CXX_STANDARD 17)
ELSE()
    # Using C++23 for host code
    SET(CMAKE_CXX_STANDARD 23)
ENDIF()
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug) # Use standard casing Debug/Release etc.

if(ENABLE_CUDA)
    message(STATUS "CUDA is enabled. Configuring CUDA support...")

    # === ADDED: Explicitly set the CUDA compiler (nvcc) path ===
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.0/bin/nvcc")
    message(STATUS "Setting CUDA compiler (nvcc) to ${CMAKE_CUDA_COMPILER}")
    # === END ADDED SECTION ===

    # === ADDED: Explicitly set the host compiler nvcc should use ===
    # Replace /usr/bin/g++-12 with the actual path to your CUDA-compatible g++
    # (e.g., /usr/bin/g++-11, /usr/bin/g++-12 depending on CUDA version support)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
    message(STATUS "Setting CUDA host compiler to ${CMAKE_CUDA_HOST_COMPILER}")
    # === END ADDED SECTION ===

    # Conditionally enable the CUDA language. This must be done after project().
    enable_language(CUDA)

    # Find the CUDA toolkit using the modern find_package approach
    find_package(CUDA REQUIRED)
    # Add a check to confirm CUDA was found
    if(CUDA_FOUND)
        message(STATUS "Found CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR} (version ${CUDA_VERSION_STRING})")
    else()
        message(FATAL_ERROR "CMake could not find the CUDA Toolkit. Please ensure it is installed and CUDACXX is in your PATH.")
    endif()

    # Reference: https://stackoverflow.com/a/43389168/6667035
    include_directories("${CUDA_INCLUDE_DIRS}")

    # Specify the C++ standard for CUDA *device* code
    # NOTE: Set based on toolkit version for broader compatibility.
    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.6") # Basic C++20 support started around 11.6
        set(CMAKE_CUDA_STANDARD 20)
    elseif(CUDA_VERSION VERSION_GREATER_EQUAL "11.0") # Basic C++17 support started around 11.0
        set(CMAKE_CUDA_STANDARD 17)
    else()
        set(CMAKE_CUDA_STANDARD 14) # Default to C++14 for older toolkits
    endif()
    message(STATUS "Setting CUDA Device C++ Standard to: ${CMAKE_CUDA_STANDARD}")
    set(CMAKE_CUDA_STANDARD_REQUIRED True)

    # Set the target GPU architecture(s).
    # sm_75 corresponds to the Turing architecture (e.g., RTX 20-series).
    # Add other architectures if needed, e.g., set(CMAKE_CUDA_ARCHITECTURES 75 86)
    set(CMAKE_CUDA_ARCHITECTURES 75)

    # --- Set CUDA Compiler Flags ---
    # These flags are passed directly to nvcc for device compilation
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

    # Add other common nvcc flags. --use_fast_math is good with -O3. For coroutines, -fcoroutines is needed.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -fcoroutines")

    # -G is for device-side debug info.
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
        message(STATUS "Enabling CUDA device debugging (-G)")
    endif()
    # Note: Flags from CMAKE_CXX_FLAGS (like -fopenmp) are automatically
    # passed to nvcc's host compiler (e.g., as -Xcompiler -fopenmp).
    # --- End CUDA Compiler Flags Section ---

else() # If CUDA is not enabled, ensure CXX settings are still applied
    message(STATUS "CUDA is disabled.")
    # CXX Standard and Build Type already set above
endif()
# --- End CUDA Setup ---


## Reference: https://stackoverflow.com/a/45934279/6667035
if(APPLE)           # Reference: https://stackoverflow.com/a/40152725/6667035
    set(CMAKE_C_COMPILER "/usr/local/bin/gcc")
    set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-15")
    # Reference: https://developer.apple.com/forums/thread/737707
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
endif()

if(UNIX AND NOT APPLE)
    # for Linux, BSD, Solaris, Minix
    set(CMAKE_C_COMPILER "/usr/bin/gcc")
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
endif()

# Set CXX flags based on compiler
# Reference: https://stackoverflow.com/a/10055571/6667035
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fopenmp") # Match standard with CMAKE_CXX_STANDARD
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  # Note: -std=c++2b is C++23. Already handled by CMAKE_CXX_STANDARD.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors -fopenmp -O3 -fdiagnostics-show-template-tree -fstats -finstrument-functions")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /Ox /Ob2 /Oi /Ot /GT /std:c++latest /experimental:module /Zm2000 /Zc:twoPhase- /D_HAS_STD_BYTE=0 /D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING /D_HAS_STD_BOOLEAN=0 /bigobj /openmp:llvm")
endif()

# Append C++ standard flag if not automatically added by CMake (older versions might need this)
# Modern CMake (>=3.1) usually adds the flag based on CMAKE_CXX_STANDARD automatically.
# Check your CMake version and compiler documentation if needed. Example for GCC/Clang:
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")


# OpenMP part
# Reference: https://stackoverflow.com/a/12404666/6667035
find_package(OpenMP)
if (OPENMP_FOUND)
    message(STATUS "Found OpenMP, applying flags.")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
     message(WARNING "OpenMP not found. Parallel CPU execution might be limited.")
endif()

# TBB part
find_package(TBB REQUIRED)
message(STATUS "Found TBB: ${TBB_INCLUDE_DIRS} | ${TBB_LIBRARIES}")

if(APPLE)
  if(IS_DIRECTORY "/opt/homebrew/lib")
    link_directories("/opt/homebrew/lib")
    message(STATUS "Manually added TBB link directory: /opt/homebrew/lib")
  elseif(IS_DIRECTORY "/usr/local/lib")
    link_directories("/usr/local/lib")
    message(STATUS "Manually added TBB link directory: /usr/local/lib")
  else()
    message(WARNING "Could not find Homebrew TBB library directory. Build may fail.")
  endif()
endif()

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

if (ENABLE_ASAN)
    message(STATUS "AddressSanitizer enabled.")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/fsanitize=address)
        # AddressSanitizer doesn't work with Edit and Continue (/ZI)
        # set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<CONFIG:Debug>,/Zi,/Z7>") # Example for controlling debug info
    elseif(NOT APPLE) # GCC/Clang and not apple
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
endif()

# add the executable
add_executable(TinyDIP main.cpp image_io.cpp)
add_executable(absTest test/absTest.cpp image_io.cpp)
add_executable(allOfTest test/allOfTest.cpp image_io.cpp)
add_executable(bicubicInterpolationTest test/bicubicInterpolationTest.cpp image_io.cpp)
add_executable(bilateralFilterTest test/bilateralFilterTest.cpp image_io.cpp)
add_executable(concatHorizontalTest test/concatHorizontalTest.cpp image_io.cpp)
add_executable(concatVerticalTest test/concatVerticalTest.cpp image_io.cpp)
add_executable(concatTest test/concatTest.cpp image_io.cpp)
add_executable(conv2Test test/conv2Test.cpp image_io.cpp)
add_executable(conv2Test2 test/conv2Test2.cpp image_io.cpp)
add_executable(convnTest test/convnTest.cpp image_io.cpp)
add_executable(DiamondWheelAnalysis test/DiamondWheelAnalysis.cpp image_io.cpp)
add_executable(dct2Test test/dct2Test.cpp image_io.cpp)
add_executable(dct2Test2 test/dct2Test2.cpp image_io.cpp)
add_executable(dct2Test3 test/dct2Test3.cpp image_io.cpp)
add_executable(dct2Test4 test/dct2Test4.cpp image_io.cpp)
add_executable(dft2Test test/dft2Test.cpp image_io.cpp)
add_executable(differenceOfGaussianTest test/differenceOfGaussianTest.cpp image_io.cpp)
add_executable(doubleImageIOTest test/doubleImageIOTest.cpp image_io.cpp)
#add_executable(dictionaryBasedNonLocalMeanTest test/dictionaryBasedNonLocalMeanTest.cpp image_io.cpp)
add_executable(gaussianFigure2DTest test/gaussianFigure2DTest.cpp image_io.cpp)
add_executable(getNFromVariadicTemplatesTest test/getNFromVariadicTemplatesTest.cpp image_io.cpp)
add_executable(highlightRegionTest test/highlightRegionTest.cpp image_io.cpp)
add_executable(histogramTest test/histogramTest.cpp image_io.cpp)
add_executable(idct2Test test/idct2Test.cpp image_io.cpp)
add_executable(imgaussfiltTest test/imgaussfiltTest.cpp image_io.cpp)
add_executable(increaseIntensityTest test/increaseIntensityTest.cpp image_io.cpp)
add_executable(isIntegerTest test/isIntegerTest.cpp image_io.cpp)
add_executable(juliaFractalGeneratorTest test/juliaFractalGeneratorTest.cpp image_io.cpp)
add_executable(lanczosResamplingTest test/lanczosResamplingTest.cpp image_io.cpp)
add_executable(mandelbrotFractalGeneratorTest test/mandelbrotFractalGeneratorTest.cpp image_io.cpp)
add_executable(manhattanDistanceTest test/manhattanDistanceTest.cpp image_io.cpp)
#add_executable(meanTest test/meanTest.cpp image_io.cpp)
add_executable(multipliesParallelTest test/multipliesParallelTest.cpp image_io.cpp)
add_executable(multipliesWithImageVectorsTest test/multipliesWithImageVectorsTest.cpp image_io.cpp)
add_executable(otsuThresholdTest test/otsuThresholdTest.cpp image_io.cpp)
add_executable(paste2DTest test/paste2DTest.cpp image_io.cpp)
add_executable(phoenixFractalGeneratorTest test/phoenixFractalGeneratorTest.cpp image_io.cpp)
add_executable(plusTest test/plusTest.cpp image_io.cpp)
add_executable(powTest test/powTest.cpp image_io.cpp)
#add_executable(powParallelTest test/powParallelTest.cpp image_io.cpp)
add_executable(printDbmpFileTest test/printDbmpFileTest.cpp image_io.cpp)
add_executable(subimageTest test/subimageTest.cpp image_io.cpp)
add_executable(subimage2Test test/subimage2Test.cpp image_io.cpp)
add_executable(rgb2hsvTest test/rgb2hsvTest.cpp image_io.cpp)
add_executable(randFunctionTest test/randFunctionTest.cpp image_io.cpp)
add_executable(dct3DetailTest test/dct3DetailTest.cpp image_io.cpp)
add_executable(dct3Test test/dct3Test.cpp image_io.cpp)
add_executable(idct3DetailTest test/idct3DetailTest.cpp image_io.cpp)
add_executable(pixelwiseOperationTest test/pixelwiseOperationTest.cpp image_io.cpp)
#add_executable(pixelwiseOperationParallelTest test/pixelwiseOperationParallelTest.cpp image_io.cpp)
add_executable(rainDictionaryAnalysis test/rainDictionaryAnalysis.cpp image_io.cpp)
add_executable(RANSAC_InlierVisualizationTest test/RANSAC_InlierVisualizationTest.cpp image_io.cpp)
add_executable(recursiveAllOfTest test/recursiveAllOfTest.cpp image_io.cpp)
add_executable(recursiveAnyOfTest test/recursiveAnyOfTest.cpp image_io.cpp)
add_executable(recursiveFindIfTest test/recursiveFindIfTest.cpp image_io.cpp)
add_executable(recursiveFoldRightAllTest test/recursiveFoldRightAllTest.cpp image_io.cpp)
add_executable(recursiveForeachTest test/recursiveForeachTest.cpp image_io.cpp)
add_executable(recursiveMinmaxTest test/recursiveMinmaxTest.cpp image_io.cpp)
add_executable(recursiveNoneOfTest test/recursiveNoneOfTest.cpp image_io.cpp)
add_executable(recursiveReduceTest test/recursiveReduceTest.cpp image_io.cpp)
add_executable(recursiveRemoveCopyTest test/recursiveRemoveCopyTest.cpp image_io.cpp)
add_executable(recursiveRemoveCopyIfTest test/recursiveRemoveCopyIfTest.cpp image_io.cpp)
add_executable(recursiveSizeTest test/recursiveSizeTest.cpp image_io.cpp)
#add_executable(recursiveTransformTest test/recursiveTransformTest.cpp image_io.cpp)
#add_executable(recursiveTransformReduceTest test/recursiveTransformReduceTest.cpp image_io.cpp)
add_executable(refineHomographyFunctionTest test/refineHomographyFunctionTest.cpp image_io.cpp)
add_executable(siftDrawMatches test/siftDrawMatches.cpp image_io.cpp)
add_executable(stitchingTest test/stitchingTest.cpp image_io.cpp)
add_executable(stitchingMultipleTest test/stitchingMultipleTest.cpp image_io.cpp)
#add_executable(sumTest test/sumTest.cpp image_io.cpp)
add_executable(testingRANSAC_MSAC_HomographyEstimation test/testingRANSAC_MSAC_HomographyEstimation.cpp image_io.cpp)
add_executable(transposeTest test/transposeTest.cpp image_io.cpp)
add_executable(windowedFilterTest test/windowedFilterTest.cpp image_io.cpp)

if(ENABLE_CUDA)
    # === NEW EXECUTABLE FOR CUDA TEST ===
    # CMake knows to compile .cu files with nvcc because CUDA was enabled.
    add_executable(stitchingCudaTest test/stitchingCudaTest.cpp image_io.cpp image_operations_cuda.cu)

    add_executable(warpPerspectiveCudaTest test/warpPerspectiveCudaTest.cpp image_io.cpp image_operations_cuda.cu)

    add_executable(warpPerspectiveCudaTest2 test/warpPerspectiveCudaTest2.cpp image_io.cpp image_operations_cuda.cu)
endif()


# target_include_directories
# Add include directories needed by ALL targets (like TBB)
# target_include_directories(TinyDIP PUBLIC ${TBB_INCLUDE_DIRS}) # Example if TBB needed includes universally

if(APPLE)
    target_include_directories(TinyDIP PRIVATE ${LLVM_INCLUDE_DIRS})
endif()

# target_link_libraries
# Reference: https://stackoverflow.com/a/39600062/6667035
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  target_link_libraries(TinyDIP stdc++fs)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  target_link_libraries(TinyDIP stdc++fs)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
endif()
target_link_libraries(TinyDIP -lm)
#target_link_libraries(TinyDIP -lomp)
target_link_libraries(TinyDIP -lpthread)
target_link_libraries(TinyDIP tbb)
target_link_libraries(absTest PUBLIC TBB::tbb)
target_link_libraries(allOfTest PUBLIC TBB::tbb)
target_link_libraries(bicubicInterpolationTest PUBLIC TBB::tbb)
target_link_libraries(bilateralFilterTest PUBLIC TBB::tbb)
target_link_libraries(concatHorizontalTest PUBLIC TBB::tbb)
target_link_libraries(concatTest PUBLIC TBB::tbb)
target_link_libraries(concatVerticalTest PUBLIC TBB::tbb)
target_link_libraries(conv2Test PUBLIC TBB::tbb)
target_link_libraries(conv2Test2 PUBLIC TBB::tbb)
target_link_libraries(convnTest PUBLIC TBB::tbb)
target_link_libraries(dct2Test PUBLIC TBB::tbb)
target_link_libraries(dct2Test2 PUBLIC TBB::tbb)
target_link_libraries(dct2Test3 PUBLIC TBB::tbb)
target_link_libraries(dct2Test4 PUBLIC TBB::tbb)
target_link_libraries(dft2Test PUBLIC TBB::tbb)
target_link_libraries(DiamondWheelAnalysis PUBLIC TBB::tbb)
target_link_libraries(differenceOfGaussianTest PUBLIC TBB::tbb)
target_link_libraries(doubleImageIOTest PUBLIC TBB::tbb)
#target_link_libraries(dictionaryBasedNonLocalMeanTest PUBLIC TBB::tbb)
target_link_libraries(gaussianFigure2DTest PUBLIC TBB::tbb)
target_link_libraries(getNFromVariadicTemplatesTest PUBLIC TBB::tbb)
target_link_libraries(highlightRegionTest PUBLIC TBB::tbb)
target_link_libraries(histogramTest PUBLIC TBB::tbb)
target_link_libraries(idct2Test PUBLIC TBB::tbb)
target_link_libraries(imgaussfiltTest PUBLIC TBB::tbb)
target_link_libraries(increaseIntensityTest PUBLIC TBB::tbb)
target_link_libraries(isIntegerTest PUBLIC TBB::tbb)
target_link_libraries(juliaFractalGeneratorTest PUBLIC TBB::tbb)
target_link_libraries(lanczosResamplingTest PUBLIC TBB::tbb)
target_link_libraries(mandelbrotFractalGeneratorTest PUBLIC TBB::tbb)
target_link_libraries(manhattanDistanceTest PUBLIC TBB::tbb)
#target_link_libraries(meanTest PUBLIC TBB::tbb)
target_link_libraries(multipliesParallelTest PUBLIC TBB::tbb)
target_link_libraries(multipliesWithImageVectorsTest PUBLIC TBB::tbb)
target_link_libraries(otsuThresholdTest PUBLIC TBB::tbb)
target_link_libraries(paste2DTest PUBLIC TBB::tbb)
target_link_libraries(phoenixFractalGeneratorTest PUBLIC TBB::tbb)
target_link_libraries(plusTest PUBLIC TBB::tbb)
target_link_libraries(powTest PUBLIC TBB::tbb)
#target_link_libraries(powParallelTest PUBLIC TBB::tbb)
target_link_libraries(printDbmpFileTest PUBLIC TBB::tbb)
target_link_libraries(subimageTest PUBLIC TBB::tbb)
target_link_libraries(subimage2Test PUBLIC TBB::tbb)
target_link_libraries(rgb2hsvTest PUBLIC TBB::tbb)
target_link_libraries(randFunctionTest PUBLIC TBB::tbb)
target_link_libraries(dct3DetailTest PUBLIC TBB::tbb)
target_link_libraries(dct3Test PUBLIC TBB::tbb)
target_link_libraries(idct3DetailTest PUBLIC TBB::tbb)
target_link_libraries(pixelwiseOperationTest PUBLIC TBB::tbb)
#target_link_libraries(pixelwiseOperationParallelTest PUBLIC TBB::tbb)
target_link_libraries(rainDictionaryAnalysis PUBLIC TBB::tbb)
target_link_libraries(RANSAC_InlierVisualizationTest PUBLIC TBB::tbb)
target_link_libraries(recursiveAllOfTest PUBLIC TBB::tbb)
target_link_libraries(recursiveAnyOfTest PUBLIC TBB::tbb)
target_link_libraries(recursiveFindIfTest PUBLIC TBB::tbb)
target_link_libraries(recursiveFoldRightAllTest PUBLIC TBB::tbb)
target_link_libraries(recursiveForeachTest PUBLIC TBB::tbb)
target_link_libraries(recursiveMinmaxTest PUBLIC TBB::tbb)
target_link_libraries(recursiveNoneOfTest PUBLIC TBB::tbb)
target_link_libraries(recursiveReduceTest PUBLIC TBB::tbb)
target_link_libraries(recursiveRemoveCopyTest PUBLIC TBB::tbb)
target_link_libraries(recursiveRemoveCopyIfTest PUBLIC TBB::tbb)
target_link_libraries(recursiveSizeTest PUBLIC TBB::tbb)
#target_link_libraries(recursiveTransformTest PUBLIC TBB::tbb)
#target_link_libraries(recursiveTransformReduceTest PUBLIC TBB::tbb)
target_link_libraries(refineHomographyFunctionTest PUBLIC TBB::tbb)
target_link_libraries(siftDrawMatches PUBLIC TBB::tbb)
target_link_libraries(stitchingTest PUBLIC TBB::tbb)
target_link_libraries(stitchingMultipleTest PUBLIC TBB::tbb)
#target_link_libraries(sumTest PUBLIC TBB::tbb)
target_link_libraries(testingRANSAC_MSAC_HomographyEstimation PUBLIC TBB::tbb)
target_link_libraries(transposeTest PUBLIC TBB::tbb)
target_link_libraries(windowedFilterTest PUBLIC TBB::tbb)

if(ENABLE_CUDA)
    # === LINK LIBRARIES FOR CUDA TEST ===
    target_link_libraries(stitchingCudaTest PUBLIC TBB::tbb)

    set_target_properties(stitchingCudaTest PROPERTIES
        CUDA_HOST_COMPILER_STANDARD ${CMAKE_CXX_STANDARD}
        CUDA_HOST_COMPILER_STANDARD_REQUIRED YES
    )
    message(STATUS "Setting CUDA Host C++ Standard for stitchingCudaTest to: ${CMAKE_CXX_STANDARD}")

    # Link against TBB like other tests
    target_link_libraries(warpPerspectiveCudaTest PUBLIC TBB::tbb)
    
    # === FIX: Set Host Compilation Standard for CUDA Target ===
    # Explicitly tell CMake to use the project's C++ standard (CMAKE_CXX_STANDARD)
    # for the host compiler when compiling CUDA code for this target.
    # This ensures headers like <format> are found correctly.
    set_target_properties(warpPerspectiveCudaTest PROPERTIES
        CUDA_HOST_COMPILER_STANDARD ${CMAKE_CXX_STANDARD}
        CUDA_HOST_COMPILER_STANDARD_REQUIRED YES
    )
    message(STATUS "Setting CUDA Host C++ Standard for warpPerspectiveCudaTest to: ${CMAKE_CXX_STANDARD}")

    target_link_libraries(warpPerspectiveCudaTest2 PUBLIC TBB::tbb)
    set_target_properties(warpPerspectiveCudaTest2 PROPERTIES
        CUDA_HOST_COMPILER_STANDARD ${CMAKE_CXX_STANDARD}
        CUDA_HOST_COMPILER_STANDARD_REQUIRED YES
    )
    message(STATUS "Setting CUDA Host C++ Standard for warpPerspectiveCudaTest2 to: ${CMAKE_CXX_STANDARD}")
endif()


if(APPLE)
    string(REPLACE "-Xclang" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Add boost library
if  (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Reference: https://github.com/Microsoft/vcpkg/issues/4188
    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_MULTITHREADED OFF)
    set(Boost_USE_STATIC_RUNTIME OFF)
    find_package(BOOST 76.0 REQUIRED COMPONENTS iterator serialization test)
    find_path(BOOST_INCLUDE_DIR boost)
    include_directories(${BOOST_INCLUDE_DIR})

    target_link_libraries(TinyDIP Boost::boost ${BOOST_LIBRARIES})
endif()

# Add opencv library
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( TinyDIP ${OpenCV_LIBS} )

# Reference: https://stackoverflow.com/a/31390758/6667035
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    list(SORT _variableNames) # Sort for easier reading
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

# Use the macro only if CMake version supports cmake_language and build type is Debug
IF (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18" AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  cmake_language(CALL print_all_variables)
ENDIF()

include(GNUInstallDirs)

install(TARGETS TinyDIP
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

