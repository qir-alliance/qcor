add_subdirectory(hybrid)

# Install QCOR standard library header
file(GLOB LIB_HEADERS qcor_*)
install(FILES ${LIB_HEADERS} DESTINATION include/qcor)
file(GLOB IMPL_HEADERS impl/*.hpp)
install(FILES ${IMPL_HEADERS} DESTINATION include/qcor/impl)
