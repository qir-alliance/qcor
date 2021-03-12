# Copy QCOR target file for Q#->QIR compilation
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/QirCore.qs DESTINATION include/qsharp)
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/QirTarget.qs DESTINATION include/qsharp)