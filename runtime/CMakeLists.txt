set(LIBRARY_NAME qcor)

file(GLOB SRC *.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME} PUBLIC . qrt)

<<<<<<< HEAD
target_link_libraries(${LIBRARY_NAME} PUBLIC xacc::xacc xacc::quantum_gate qrt xacc::pauli)
=======
target_link_libraries(${LIBRARY_NAME} PUBLIC xacc::xacc xacc::quantum_gate qrt xacc::pauli xacc::fermion)
>>>>>>> init commit

xacc_configure_library_rpath(${LIBRARY_NAME})

file(GLOB HEADERS qcor.hpp)
install(FILES ${HEADERS} DESTINATION include/qcor)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

if (QCOR_BUILD_TESTS)
  add_subdirectory(tests)
endif()

add_subdirectory(objectives)
add_subdirectory(qrt)