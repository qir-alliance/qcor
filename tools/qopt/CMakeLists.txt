
# Store the location of the clang executable
set (OPT_EXECUTABLE "${LLVM_INSTALL_PREFIX}/bin/opt")

configure_file(qopt.in
               ${CMAKE_BINARY_DIR}/qopt)

add_subdirectory(passes)

install(PROGRAMS ${CMAKE_BINARY_DIR}/qopt DESTINATION bin)
