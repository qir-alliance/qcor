
# Store the location of the clang executable
set (CLANG_EXECUTABLE "${LLVM_INSTALL_PREFIX}/bin/clang++")
set(QCOR_APPEND_PLUGIN_PATH "")
if (NOT ${XACC_ROOT} MATCHES ${CMAKE_INSTALL_PREFIX}) 
  set (QCOR_APPEND_PLUGIN_PATH "${CMAKE_INSTALL_PREFIX}/plugins")
endif()

configure_file(qcor.in
               ${CMAKE_BINARY_DIR}/qcor)

install(PROGRAMS ${CMAKE_BINARY_DIR}/qcor DESTINATION bin)
