


link_directories(${XACC_ROOT}/lib)
add_executable(qasm3VisitorTester test_quantum_declaration.cpp)
add_test(NAME qcor_qasm3_quantum_decl_tester COMMAND qasm3VisitorTester)
target_include_directories(qasm3VisitorTester PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3VisitorTester qcor-mlir-api gtest gtest_main)

