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
set(LIBRARY_NAME qcor-ftqc-qrt)

file(GLOB SRC *.cpp tracer/*.cpp)

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC . ../..)

target_link_libraries(${LIBRARY_NAME} PUBLIC qcor CppMicroServices::CppMicroServices)

set(_bundle_name qcor_ftqc_qrt)
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
                        PROPERTIES INSTALL_RPATH "@loader_path/../lib;${XACC_ROOT}/lib")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "$ORIGIN/../lib:${XACC_ROOT}/lib")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

if (QCOR_BUILD_TESTS) 
  add_subdirectory(tests)
endif()

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)
