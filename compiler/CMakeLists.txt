add_subdirectory(clang)

set(LIBRARY_NAME qcor-compiler)

file(GLOB SRC *.cpp)

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME}
                            PUBLIC .
                                ${XACC_INCLUDE_DIRS}
                                ${CLANG_INCLUDE_DIRS}
                                ${LLVM_INCLUDE_DIRS}
                                )

target_link_libraries(${LIBRARY_NAME} PUBLIC xacc qcor-ast-plugin ${CLANG_LIBS} ${LLVM_LIBS})

set(_bundle_name qcor_compiler)
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

qcor_enable_rpath(${LIBRARY_NAME})

if(QCOR_BUILD_TESTS)
  #add_subdirectory(tests)
endif()

file(GLOB HEADERS *.hpp)

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)
