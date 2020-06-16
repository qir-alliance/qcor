
add_library(xacc_jit_pass MODULE
    # List your source files here.
    xacc_jit_pass.cpp
)

target_include_directories(xacc_circuit_optimizer PRIVATE ${LLVM_INCLUDE_DIRS})
# Use C++11 to compile our pass (i.e., supply -std=c++11).
target_compile_features(xacc_jit_pass PRIVATE cxx_range_for cxx_auto_type)

# LLVM is (typically) built with no C++ RTTI. We need to match that;
# otherwise, we'll get linker errors about missing RTTI data.
set_target_properties(xacc_jit_pass PROPERTIES
    COMPILE_FLAGS "-fno-rtti"
)

target_include_directories(xacc_jit_pass PRIVATE .. ../xacc-ir-transformation ${LLVM_INCLUDE_DIRS})
target_link_libraries(xacc_jit_pass
                      PRIVATE ${LLVM_LIBS} xacc::xacc xacc::quantum_gate qrt)

# Get proper shared-library behavior (where symbols are not necessarily
# resolved when the shared library is linked) on OS X.
if(APPLE)
    set_target_properties(print_llvm_qir PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif(APPLE)


if(APPLE)
  set_target_properties(xacc_jit_pass
                        PROPERTIES INSTALL_RPATH "@loader_path/../lib;${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(xacc_circuit_optimizer
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(xacc_jit_pass
                        PROPERTIES INSTALL_RPATH "$ORIGIN/../lib:${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(xacc_circuit_optimizer PROPERTIES LINK_FLAGS "-shared")
endif()


install(TARGETS xacc_jit_pass DESTINATION qopt-plugins)
