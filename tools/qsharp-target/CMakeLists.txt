# Copy QCOR target file for Q#->QIR compilation
file(GLOB QSHARP_TARGET_CONFIGS *.qs)

foreach(ConfigFile ${QSHARP_TARGET_CONFIGS})
    configure_file(${ConfigFile} ${CMAKE_BINARY_DIR}/${ConfigFile})
    install(PROGRAMS ${CMAKE_BINARY_DIR}/${ConfigFile} DESTINATION include/qsharp)
endforeach()