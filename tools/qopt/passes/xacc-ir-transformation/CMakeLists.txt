set(LIBRARY_NAME xacc-llvm-compiler)

set(SRC xacc_llvm_compiler.cpp)

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME} PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/runtime/qrt)

# Use C++11 to compile our pass (i.e., supply -std=c++11).
target_compile_features(${LIBRARY_NAME} PRIVATE cxx_range_for cxx_auto_type)


set(_bundle_name xacc_llvm_compiler)
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

target_link_libraries(${LIBRARY_NAME}
                      PRIVATE xacc::xacc xacc::quantum_gate qcor-clang-wrapper)

install(TARGETS ${LIBRARY_NAME} DESTINATION plugins)

if (QCOR_BUILD_TESTS)
add_subdirectory(tests)
endif()

if(APPLE)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "@loader_path/../lib")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()


add_library(xacc_circuit_optimizer MODULE
    # List your source files here.
    xacc_circuit_optimizer.cpp
)

target_include_directories(xacc_circuit_optimizer PRIVATE ${LLVM_INCLUDE_DIRS})
# Use C++11 to compile our pass (i.e., supply -std=c++11).
target_compile_features(xacc_circuit_optimizer PRIVATE cxx_range_for cxx_auto_type)

# LLVM is (typically) built with no C++ RTTI. We need to match that;
# otherwise, we'll get linker errors about missing RTTI data.
set_target_properties(xacc_circuit_optimizer PROPERTIES
    COMPILE_FLAGS "-fno-rtti"
)

target_include_directories(xacc_circuit_optimizer PRIVATE ..)
target_link_libraries(xacc_circuit_optimizer
                      PRIVATE ${LLVM_LIBS} xacc::xacc xacc::quantum_gate)

# Get proper shared-library behavior (where symbols are not necessarily
# resolved when the shared library is linked) on OS X.
if(APPLE)
    set_target_properties(print_llvm_qir PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif(APPLE)


if(APPLE)
  set_target_properties(xacc_circuit_optimizer
                        PROPERTIES INSTALL_RPATH "@loader_path/../lib;${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(xacc_circuit_optimizer
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(xacc_circuit_optimizer
                        PROPERTIES INSTALL_RPATH "$ORIGIN/../lib:${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(xacc_circuit_optimizer PROPERTIES LINK_FLAGS "-shared")
endif()


install(TARGETS xacc_circuit_optimizer DESTINATION qopt-plugins)
