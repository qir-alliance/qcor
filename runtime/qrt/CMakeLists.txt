set(LIBRARY_NAME qrt)

file(GLOB SRC *.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME} PUBLIC . ${XACC_ROOT}/include/eigen)

target_link_libraries(${LIBRARY_NAME} PUBLIC xacc::xacc xacc::quantum_gate xacc::pauli)

xacc_configure_library_rpath(${LIBRARY_NAME})

file(GLOB HEADERS qrt.hpp)
install(FILES ${HEADERS} DESTINATION include/qcor)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

add_subdirectory(impls)