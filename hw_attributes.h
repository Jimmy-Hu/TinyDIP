#pragma once

// Abstracting hardware compiler specific annotations.
// These macros ensure compatibility with non-Clang compilers (e.g., GCC in GitHub CI)
// while providing crucial MLIR generation hints for CIRCT/Polygeist.

#if defined(__clang__)
    #define TINYDIP_HW_FLATTEN [[clang::annotate("circt.hw.flatten")]]
    #define TINYDIP_HW_AXILITE [[clang::annotate("circt.hw.axilite_register")]]
#else
    // Expands to nothing when compiled with GCC or MSVC
    #define TINYDIP_HW_FLATTEN
    #define TINYDIP_HW_AXILITE
#endif
