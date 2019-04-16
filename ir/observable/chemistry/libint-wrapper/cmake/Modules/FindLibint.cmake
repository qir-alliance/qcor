# - Try to find LibInt
#
#  In order to aid find_package the user may set LIBINT_ROOT_DIR to the root of
#  the installed libint.
#
#  Once done this will define
#  LIBINT_FOUND - System has Libint
#  LIBINT_INCLUDE_DIRS - The Libint include directories
#  LIBINT_LIBRARIES - The libraries needed to use Libint

#Prefer LIBINT_ROOT_DIR if the user specified it
if(NOT DEFINED LIBINT_ROOT_DIR)
    find_package(PkgConfig)
    pkg_check_modules(PC_LIBINT libint2)
endif()

find_path(LIBINT_INCLUDE_DIR libint2.hpp
          HINTS ${PC_LIBINT_INCLUDEDIR}
                ${PC_LIBINT_INCLUDE_DIRS}
          PATHS ${LIBINT_ROOT_DIR}/include
          PATH_SUFFIXES libint)

find_library(LIBINT_LIBRARY NAMES libint2${CMAKE_STATIC_LIBRARY_SUFFIX}
             HINTS ${PC_LIBINT_LIBDIR}
                   ${PC_LIBINT_LIBRARY_DIRS}
             PATHS ${LIBINT_ROOT_DIR}/lib
        )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBINT DEFAULT_MSG
                                  LIBINT_LIBRARY LIBINT_INCLUDE_DIR)

set(Libint_FOUND ${LIBINT_FOUND})
set(LIBINT_LIBRARIES ${LIBINT_LIBRARY})
set(LIBINT_INCLUDE_DIRS ${LIBINT_INCLUDE_DIR})
