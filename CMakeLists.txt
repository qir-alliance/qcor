cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

if (NOT XACC_DIR)
  set(XACC_DIR "$ENV{HOME}/.xacc")
endif()

set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_DISABLE_IN_SOURCE_BUILDS ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/Modules)

project(qcor LANGUAGES CXX)

option(QCOR_BUILD_TESTS "Build qcor tests" OFF)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD WORKING_DIRECTORY "${local_dir}"
                            OUTPUT_VARIABLE QCOR_BUILD_VERSION ERROR_QUIET
                            OUTPUT_STRIP_TRAILING_WHITESPACE)
  message( STATUS "QCOR GIT hash: ${QCOR_BUILD_VERSION}")
endif()

find_package(Clang 10.0.0 REQUIRED)
find_package(XACC REQUIRED)

configure_file("${CMAKE_SOURCE_DIR}/cmake/qcor_config.hpp.in"
               "${CMAKE_BINARY_DIR}/qcor_config.hpp")
install (FILES ${CMAKE_BINARY_DIR}/qcor_config.hpp DESTINATION include/qcor)

add_subdirectory(handlers)
add_subdirectory(runtime)
add_subdirectory(tools)
add_subdirectory(lib)

if (QCOR_BUILD_TESTS)
  add_subdirectory(examples)
endif()

# Compile and install Python binding if Python found:
find_package(Python COMPONENTS Interpreter Development)
if(Python_FOUND)
 if(${Python_VERSION} VERSION_GREATER_EQUAL 3.0.0)
   message(STATUS "Found Python version ${Python_VERSION}. Building QCOR Python API with ${Python_INCLUDE_DIRS}")
   add_subdirectory(python)
 else()
  message(STATUS "Found Python version ${Python_VERSION}. Version must be greater than 3.0.0, skipping Python API build.")
 endif()
else()
  message(STATUS "Python interpreter or development headers not found. Skipping Python API build.")
endif()

if (QCOR_CPACK_DEB_PLATFORM)
message(STATUS "CPack DEB Build Enabled.") 
set(CPACK_SET_DESTDIR "on")
set(CPACK_PACKAGING_INSTALL_PREFIX "/tmp")
set(CPACK_GENERATOR "DEB")

set(MAJOR_VERSION 1)
set(MINOR_VERSION 0)
set(PATCH_VERSION 0)

set(CPACK_PACKAGE_DESCRIPTION "qcor quantum-classical c++ compiler")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "qcor is a c++ compiler for heterogenous quantum-classical computing")
set(CPACK_PACKAGE_VENDOR "ORNL")
set(CPACK_PACKAGE_CONTACT "mccaskeyaj@ornl.gov")
set(CPACK_PACKAGE_VERSION_MAJOR "${MAJOR_VERSION}")
set(CPACK_PACKAGE_VERSION_MINOR "${MINOR_VERSION}")
set(CPACK_PACKAGE_VERSION_PATCH "${PATCH_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_SOURCE_DIR}/scripts/debian/postinst")

#dependencies for this service menu
if (${QCOR_CPACK_DEB_PLATFORM} STREQUAL "bionic")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "xacc, clang-syntax-handler")
elseif(${QCOR_CPACK_DEB_PLATFORM} STREQUAL "focal")
 set(CPACK_DEBIAN_PACKAGE_DEPENDS "xacc, clang-syntax-handler")
endif()
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_SECTION "base")
set(CPACK_DEBIAN_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
 
include(CPack)
endif()