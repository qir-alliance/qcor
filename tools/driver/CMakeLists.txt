
# Store the location of the clang executable
set (CLANG_EXECUTABLE "${LLVM_INSTALL_PREFIX}/bin/clang++")

if (APPLE) 
configure_file(qcor-macosx.in
               ${CMAKE_BINARY_DIR}/qcor)
else() 
configure_file(qcor.in
               ${CMAKE_BINARY_DIR}/qcor)
endif()
install(PROGRAMS ${CMAKE_BINARY_DIR}/qcor DESTINATION bin)
