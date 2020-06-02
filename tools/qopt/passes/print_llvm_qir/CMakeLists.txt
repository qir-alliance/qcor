add_library(print_llvm_qir MODULE
    # List your source files here.
    print_llvm_qir.cpp
)

target_include_directories(print_llvm_qir PRIVATE ${LLVM_INCLUDE_DIRS})
# Use C++11 to compile our pass (i.e., supply -std=c++11).
target_compile_features(print_llvm_qir PRIVATE cxx_range_for cxx_auto_type)

# LLVM is (typically) built with no C++ RTTI. We need to match that;
# otherwise, we'll get linker errors about missing RTTI data.
set_target_properties(print_llvm_qir PROPERTIES
    COMPILE_FLAGS "-fno-rtti"
)

target_include_directories(print_llvm_qir PRIVATE ..)
target_link_libraries(print_llvm_qir
                      PRIVATE ${LLVM_LIBS})

# Get proper shared-library behavior (where symbols are not necessarily
# resolved when the shared library is linked) on OS X.
if(APPLE)
    set_target_properties(print_llvm_qir PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif(APPLE)

install(TARGETS print_llvm_qir DESTINATION qopt-plugins)
