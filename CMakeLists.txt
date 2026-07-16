# Reference: https://cmake.org/cmake/help/latest/guide/tutorial/index.html
cmake_minimum_required(VERSION 3.13)        # For lgtm.com service

# set the project name and version
# Reference: https://stackoverflow.com/a/41651622/6667035
# Start with C and CXX, CUDA will be enabled conditionally
project(TinyDIP VERSION 1.0 LANGUAGES C CXX)

# print compile/link commands
# Reference: https://stackoverflow.com/a/3379246/6667035
set(CMAKE_VERBOSE_MAKEFILE on)

# --- RPATH Configuration for Custom GCC (e.g., GCC 17) ---
# Define the directory containing the new libgomp.so.1 and libstdc++.so.6
# Common locations: /usr/local/lib64, /opt/gcc-17/lib64, /opt/gcc-latest/lib64
set(CUSTOM_GCC_LIB_DIR "/usr/local/lib64" CACHE PATH "Path to custom GCC library directory")

# Use, i.e. do not skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# Use the install RPATH during the build phase as well
# This ensures that executables run directly from the build directory can find the libraries
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Set the RPATH to be used when installing
# CRITICAL PORTABILITY FIX: Added $ORIGIN/lib to enable "Option A: Bundling Shared Libraries" natively!
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib;${CUSTOM_GCC_LIB_DIR}")

# Add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# ---------------------------------------------------------

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
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.3/bin/nvcc")
    message(STATUS "Setting CUDA compiler (nvcc) to ${CMAKE_CUDA_COMPILER}")
    # === END ADDED SECTION ===

    # === ADDED: Explicitly set the host compiler nvcc should use ===
    # Replace /usr/bin/g++-12 with the actual path to your CUDA-compatible g++
    # (e.g., /usr/bin/g++-11, /usr/bin/g++-12 depending on CUDA version support)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-15)
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
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -fcoroutines -fstack-usage")

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
#if(APPLE)           # Reference: https://stackoverflow.com/a/40152725/6667035
#   set(CMAKE_C_COMPILER "/usr/local/bin/gcc")
#   set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-15")
    # Reference: https://developer.apple.com/forums/thread/737707
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
#endif()

if(UNIX AND NOT APPLE)
    # for Linux, BSD, Solaris, Minix
    # Only set specific paths if necessary, otherwise trust CMake detection
    #if(EXISTS "/usr/bin/gcc")
    #    set(CMAKE_C_COMPILER "/usr/bin/gcc")
    #endif()
    #if(EXISTS "/usr/bin/g++")
    #    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
    #endif()
endif()

# Set CXX flags based on compiler
# Reference: https://stackoverflow.com/a/10055571/6667035
# Modified: Use MATCHES "Clang" instead of STREQUAL "Clang" to correctly catch AppleClang on macOS
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # using Clang or AppleClang
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math") # Match standard with CMAKE_CXX_STANDARD
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  # Note: -std=c++2b is C++23. Already handled by CMAKE_CXX_STANDARD.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors -fopenmp -O3 -fdiagnostics-show-template-tree -fstats -fstack-usage")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
  # Removed deprecated /experimental:module
  # Added /D_CRT_SECURE_NO_WARNINGS to suppress fopen warnings
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /Ox /Ob2 /Oi /Ot /GT /std:c++latest /Zm2000 /Zc:twoPhase- /D_HAS_STD_BYTE=0 /D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING /D_HAS_STD_BOOLEAN=0 /bigobj /openmp:llvm /D_CRT_SECURE_NO_WARNINGS")
endif()

# Append C++ standard flag if not automatically added by CMake (older versions might need this)
# Modern CMake (>=3.1) usually adds the flag based on CMAKE_CXX_STANDARD automatically.
# Check your CMake version and compiler documentation if needed. Example for GCC/Clang:
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")


# OpenMP part
# Reference: https://stackoverflow.com/a/12404666/6667035

# === ADDED: Help CMake find OpenMP (libomp) on macOS (Homebrew) ===
if(APPLE)
    if(IS_DIRECTORY "/opt/homebrew/opt/libomp")
        set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
    elseif(IS_DIRECTORY "/usr/local/opt/libomp")
        set(OpenMP_ROOT "/usr/local/opt/libomp")
    endif()
endif()
# === END ADDED ===

find_package(OpenMP)
if (OPENMP_FOUND)
    message(STATUS "Found OpenMP, applying flags.")
    
    # === ADDED: Fix for AppleClang unsupported option '-fopenmp' ===
    if (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        # AppleClang requires -Xpreprocessor before -fopenmp
        # and explicit linking to libomp (usually handled by find_package in newer CMake, but explicitly fixed here just in case)
        if (NOT "${OpenMP_CXX_FLAGS}" MATCHES "-Xpreprocessor")
            message(STATUS "Patching OpenMP flags for AppleClang...")
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_EXE_LINKER_FLAGS "-lomp")
        endif()
    endif()
    # === END ADDED ===

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

# ------------------------------------------------------------------------------------
#  Test Executables (Refactored using foreach to reduce code duplication)
# ------------------------------------------------------------------------------------
set(TINYDIP_TESTS
    absTest
    allOfTest
    bicubicInterpolationTest
    bilateralFilterTest
    concatHorizontalTest
    concatVerticalTest
    concatTest
    conv2Test
    conv2Test2
    convnTest
    DiamondWheelAnalysis
    dct2Test
    dct2Test2
    dct2Test3
    dct2Test4
    dft2Test
    differenceOfGaussianTest
    doubleImageIOTest
    # dictionaryBasedNonLocalMeanTest 
    gaussianFigure2DTest
    getNFromVariadicTemplatesTest
    highlightRegionTest
    histogramTest
    idct2Test
    imgaussfiltTest
    increaseIntensityTest
    isIntegerTest
    juliaFractalGeneratorTest
    lanczosResamplingTest
    mandelbrotFractalGeneratorTest
    manhattanDistanceTest
    # meanTest 
    multipliesParallelTest
    multipliesWithImageVectorsTest
    otsuThresholdTest
    paste2DTest
    phoenixFractalGeneratorTest
    plusTest
    powTest
    # powParallelTest 
    printDbmpFileTest
    subimageTest
    subimage2Test
    rgb2hsvTest
    randFunctionTest
    dct3DetailTest
    dct3Test
    idct3DetailTest
    pixelwiseOperationTest
    # pixelwiseOperationParallelTest 
    rainDictionaryAnalysis
    RANSAC_InlierVisualizationTest
    recursiveAllOfTest
    recursiveAnyOfTest
    recursiveFindIfTest
    recursiveFoldRightAllTest
    recursiveForeachTest
    recursiveMinmaxTest
    recursiveNoneOfTest
    recursiveReduceTest
    recursiveRemoveCopyTest
    recursiveRemoveCopyIfTest
    recursiveSizeTest
    # recursiveTransformTest 
    # recursiveTransformReduceTest 
    refineHomographyFunctionTest
    siftDrawMatches
    stitchingTest
    stitchingMultipleTest
    # sumTest 
    testingRANSAC_MSAC_HomographyEstimation
    transposeTest
    windowedFilterTest
)

# Loop through all tests to generate targets and link dependencies uniformly
foreach(TEST_NAME IN LISTS TINYDIP_TESTS)
    add_executable(${TEST_NAME} test/${TEST_NAME}.cpp image_io.cpp)
    # FIX: Ensure ALL tests link against OpenCV, TBB, and other necessary libraries
    target_link_libraries(${TEST_NAME} PUBLIC TBB::tbb ${OpenCV_LIBS})
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_link_libraries(${TEST_NAME} PUBLIC stdc++fs)
    endif()
endforeach()

# target_include_directories
# Add include directories needed by ALL targets (like TBB)
# target_include_directories(TinyDIP PUBLIC ${TBB_INCLUDE_DIRS}) # Example if TBB needed includes universally

if(APPLE)
    target_include_directories(TinyDIP PRIVATE ${LLVM_INCLUDE_DIRS})
endif()

# target_link_libraries
# Reference: https://stackoverflow.com/a/39600062/6667035
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
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
target_link_libraries(TinyDIP ${CMAKE_DL_LIBS})
target_link_libraries(TinyDIP tbb)


if(APPLE)
    string(REPLACE "-Xclang" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Add boost library
if  (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Reference: https://github.com/Microsoft/vcpkg/issues/4188
    # Vcpkg default for x64-windows is Shared (Dynamic) and Multithreaded.
    # We should let Vcpkg/CMake handle the selection or explicitly match Vcpkg.
    # Removing Boost_USE_STATIC_LIBS etc. to allow automatic detection of vcpkg's shared libs.
    
    # Fix package name case (BOOST -> Boost) and remove version constraint
    # Change component 'test' to 'unit_test_framework' which is the standard name
    find_package(Boost REQUIRED COMPONENTS iterator serialization unit_test_framework)
    
    if(Boost_FOUND)
        include_directories(${Boost_INCLUDE_DIRS})
        # Link against imported targets or libraries variable
        # Note: Boost::boost is header-only, for components use Boost::component_name
        target_link_libraries(TinyDIP Boost::boost ${Boost_LIBRARIES})
    endif()
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

# ------------------------------------------------------------------------------------
#  Dynamic Plugin Architecture Configuration
# ------------------------------------------------------------------------------------

# Configure Plugin Output Directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins)

# CRITICAL FIX: Force the host executable to export its internal symbols (like bmp_read)
# to the dynamic symbol table so that loaded plugins can mathematically resolve them at runtime!
set_target_properties(TinyDIP PROPERTIES ENABLE_EXPORTS TRUE)

# Add Math Plugin Module
add_library(plugin_math MODULE plugin_math.cpp)
set_target_properties(plugin_math PROPERTIES
    PREFIX ""
    OUTPUT_NAME "tinydip_math"
)

# Ensure the plugin correctly links against TBB if multithreading handlers are invoked,
# AND explicitly link to TinyDIP to resolve host-level C++ symbols natively!
target_link_libraries(plugin_math PRIVATE tbb TinyDIP)

# Add Math (Add) Plugin Module
add_library(plugin_math_add MODULE plugin_math_add.cpp)
set_target_properties(plugin_math_add PROPERTIES
    PREFIX ""
    OUTPUT_NAME "tinydip_math_add"
)

target_link_libraries(plugin_math_add PRIVATE tbb TinyDIP)

include(GNUInstallDirs)

install(TARGETS TinyDIP
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

