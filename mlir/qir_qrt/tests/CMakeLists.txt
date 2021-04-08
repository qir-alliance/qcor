link_directories(${XACC_ROOT}/lib)
add_executable(QirQrtTester QirQrtTester.cpp)
add_test(NAME qcor_QirQrtTester COMMAND QirQrtTester)
target_include_directories(QirQrtTester PRIVATE . .. ${CMAKE_BINARY_DIR} ${XACC_ROOT}/include/gtest)
target_link_libraries(QirQrtTester ${XACC_TEST_LIBRARIES} xacc::xacc qir-qrt qrt)

# Test compile a QASM -> QIR lowering
add_test(NAME mlir_qrt_compile COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/bell_qasm.qasm)

add_executable(QirQrtRefCountTester QirQrtRefCountTester.cpp)
add_test(NAME qcor_QirQrtRefCountTester COMMAND QirQrtRefCountTester)
target_include_directories(QirQrtRefCountTester PRIVATE . .. ${CMAKE_BINARY_DIR} ${XACC_ROOT}/include/gtest)
target_link_libraries(QirQrtRefCountTester ${XACC_TEST_LIBRARIES} xacc::xacc qir-qrt qrt)
