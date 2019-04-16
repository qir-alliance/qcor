cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_DISABLE_IN_SOURCE_BUILDS ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(LIBINT_ROOT_DIR /usr/local/libint/2.5.0 CACHE PATH "LibInt Path")

project(libint-wrapper LANGUAGES CXX)

option(LIBINT_BUILD_TESTS "Build qcor tests" OFF)

include(CTest)

find_package(Libint 1.2.0)
if (NOT Libint_FOUND)
   message (FATAL_ERROR "Cannot find LibInt. Please use LIBINT_ROOT_DIR argument.")
endif()

#add_subdirectory(tpls)
add_subdirectory(src)