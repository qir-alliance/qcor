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

message(STATUS "${CMAKE_MODULE_PATH}")
project(qcor LANGUAGES CXX)

option(QCOR_BUILD_TESTS "Build qcor tests" OFF)

include(CTest)

find_package(Clang 8.0.0 REQUIRED)
find_package(XACC REQUIRED)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${XACC_ROOT}" CACHE PATH "default install path" FORCE )
endif()

set(CppMicroServicesConfig_DIR "${XACC_ROOT}/share/cppmicroservices4/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${XACC_ROOT}/share/xacc")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CppMicroServicesConfig_DIR}")
include(tests)
include(CppMicroServicesConfig)

include_directories(${XACC_INCLUDE_DIRS})

link_directories(${XACC_LIBRARY_DIR})

message(STATUS "${CLANG_INCLUDE_DIRS}")

add_subdirectory(compiler)
add_subdirectory(instructions)
