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

find_package(Clang 9.0.0 REQUIRED)
find_package(XACC REQUIRED)
set(CLANG_COMPILER /usr/bin/clang++-9)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${XACC_ROOT}" CACHE PATH "default install path" FORCE )
endif()

set(CppMicroServicesConfig_DIR "${XACC_ROOT}/share/cppmicroservices4/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${XACC_ROOT}/share/xacc")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CppMicroServicesConfig_DIR}")
include(CppMicroServicesConfig)

include_directories(${XACC_INCLUDE_DIRS})

link_directories(${XACC_LIBRARY_DIR})

configure_file(${CMAKE_SOURCE_DIR}/scripts/qcor.in
               ${CMAKE_BINARY_DIR}/qcor)

macro(qcor_enable_rpath LIBNAME)
  if(APPLE)
    set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH "@loader_path/../lib")
    set_target_properties(${LIBNAME}
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  else()
    set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
    set_target_properties(${LIBNAME} PROPERTIES LINK_FLAGS "-shared")
  endif()
endmacro()

add_subdirectory(runtime)
add_subdirectory(compiler)
add_subdirectory(ir)

install(PROGRAMS ${CMAKE_BINARY_DIR}/qcor DESTINATION bin)