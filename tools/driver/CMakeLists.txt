
# Store the location of the clang executable
set (CLANG_EXECUTABLE "${LLVM_INSTALL_PREFIX}/bin/clang++")
set(QCOR_APPEND_PLUGIN_PATH "")
if (NOT ${XACC_ROOT} MATCHES ${CMAKE_INSTALL_PREFIX}) 
  message(STATUS "WE ARE BUILDING WITH HOMEBREW")
  set (QCOR_APPEND_PLUGIN_PATH "${CMAKE_INSTALL_PREFIX}/plugins")
endif()

if (APPLE) 
configure_file(qcor-macosx.in
               ${CMAKE_BINARY_DIR}/qcor)
else() 
configure_file(qcor.in
               ${CMAKE_BINARY_DIR}/qcor)
endif()
install(PROGRAMS ${CMAKE_BINARY_DIR}/qcor DESTINATION bin)
