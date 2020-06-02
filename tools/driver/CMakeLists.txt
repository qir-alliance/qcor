
# Store the location of the clang executable
set (CLANG_EXECUTABLE "${LLVM_INSTALL_PREFIX}/bin/clang++")

configure_file(qcor.in
               ${CMAKE_BINARY_DIR}/qcor)

install(PROGRAMS ${CMAKE_BINARY_DIR}/qcor DESTINATION bin)
