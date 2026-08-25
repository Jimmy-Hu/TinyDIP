# Module to find the Polygeist compiler executable.
# 
# This module defines the following variables:
#   Polygeist_FOUND      - True if the polygeist executable was found.
#   POLYGEIST_EXECUTABLE - The absolute path to the polygeist compiler.

# Search for the executable in the system PATH and standard directories.
find_program(POLYGEIST_EXECUTABLE
    NAMES polygeist
    DOC "Path to the Polygeist compiler executable"
)

# Use standard CMake macros to handle REQUIRED and QUIET arguments,
# and set Polygeist_FOUND to TRUE if the executable is found.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Polygeist
    REQUIRED_VARS POLYGEIST_EXECUTABLE
)

if (Polygeist_FOUND AND NOT TARGET Polygeist::polygeist)
    add_executable(Polygeist::polygeist IMPORTED)
    set_target_properties(Polygeist::polygeist PROPERTIES
        IMPORTED_LOCATION "${POLYGEIST_EXECUTABLE}"
    )
endif()