# Module to find the CIRCT toolchain executables.
# 
# This module defines the following variables:
#   CIRCT_FOUND                - True if all required CIRCT executables were found.
#   CIRCT_OPT_EXECUTABLE       - The absolute path to the circt-opt executable.
#   CIRCT_TRANSLATE_EXECUTABLE - The absolute path to the circt-translate executable.

# Search for circt-opt in the system PATH and standard directories.
find_program(CIRCT_OPT_EXECUTABLE
    NAMES circt-opt
    DOC "Path to the circt-opt executable"
)

# Search for circt-translate in the system PATH and standard directories.
find_program(CIRCT_TRANSLATE_EXECUTABLE
    NAMES circt-translate
    DOC "Path to the circt-translate executable"
)

# Use standard CMake macros to handle REQUIRED and QUIET arguments.
# This will set CIRCT_FOUND to TRUE only if BOTH executables are successfully located.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CIRCT
    REQUIRED_VARS CIRCT_OPT_EXECUTABLE CIRCT_TRANSLATE_EXECUTABLE
)

# Create imported executable targets for modern CMake usage.
if (CIRCT_FOUND)
    if (NOT TARGET CIRCT::opt)
        add_executable(CIRCT::opt IMPORTED)
        set_target_properties(CIRCT::opt PROPERTIES
            IMPORTED_LOCATION "${CIRCT_OPT_EXECUTABLE}"
        )
    endif()

    if (NOT TARGET CIRCT::translate)
        add_executable(CIRCT::translate IMPORTED)
        set_target_properties(CIRCT::translate PROPERTIES
            IMPORTED_LOCATION "${CIRCT_TRANSLATE_EXECUTABLE}"
        )
    endif()
endif()