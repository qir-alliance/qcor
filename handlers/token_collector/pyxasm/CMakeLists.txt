# *******************************************************************************
# Copyright (c) 2019 UT-Battelle, LLC.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v1.0
# and Eclipse Distribution License v.10 which accompany this distribution.
# The Eclipse Public License is available at http://www.eclipse.org/legal/epl-v10.html
# and the Eclipse Distribution License is available at
# https://eclipse.org/org/documents/edl-v10.php
#
# Contributors:
#   Alexander J. McCaskey - initial API and implementation
# *******************************************************************************/
set(LIBRARY_NAME qcor-pyxasm-token)

set(ANTLR_LIB ${XACC_ROOT}/lib/libantlr4-runtime.so)
if (APPLE)
  set(ANTLR_LIB ${XACC_ROOT}/lib/libantlr4-runtime.dylib)
endif()

file(GLOB SRC *.cpp generated/*.cpp)

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_compile_options(${LIBRARY_NAME} PRIVATE "-Wno-attributes")
target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC . .. ${CMAKE_SOURCE_DIR}/runtime/utils ${XACC_ROOT}/include/antlr4-runtime generated ${CLANG_INCLUDE_DIRS})

target_link_libraries(${LIBRARY_NAME} PUBLIC ${CLANG_LIBS} ${LLVM_LIBS} ${ANTLR_LIB} qrt xacc::xacc)

set(_bundle_name qcor_pyxasm_token)
set_target_properties(${LIBRARY_NAME}
                      PROPERTIES COMPILE_DEFINITIONS
                                 US_BUNDLE_NAME=${_bundle_name}
                                 US_BUNDLE_NAME
                                 ${_bundle_name})

usfunctionembedresources(TARGET
                         ${LIBRARY_NAME}
                         WORKING_DIRECTORY
                         ${CMAKE_CURRENT_SOURCE_DIR}
                         FILES
                         manifest.json)


if(APPLE)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${XACC_ROOT}/lib;${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "$ORIGIN/../lib:${XACC_ROOT}/lib:${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)

if (QCOR_BUILD_TESTS)
add_subdirectory(tests)
endif()