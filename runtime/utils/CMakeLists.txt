set(LIBRARY_NAME utils)

file(GLOB SRC *.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME} PUBLIC .)

target_link_libraries(${LIBRARY_NAME} PUBLIC xacc::xacc xacc::quantum_gate xacc::pauli xacc::fermion)

xacc_configure_library_rpath(${LIBRARY_NAME})

file(GLOB HEADERS operator_algebra.hpp)
install(FILES ${HEADERS} DESTINATION include/qcor)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
