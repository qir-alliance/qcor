


link_directories(${XACC_ROOT}/lib)
#add_executable(qasm3VisitorTester test_quantum_declaration.cpp)
#add_test(NAME qcor_qasm3_quantum_decl_tester COMMAND qasm3VisitorTester)
#target_include_directories(qasm3VisitorTester PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
#target_link_libraries(qasm3VisitorTester qcor-mlir-api gtest gtest_main)

add_executable(qasm3CompilerTester_Assignment test_assignment.cpp)
add_test(NAME qcor_qasm3_classical_assignment_tester COMMAND qasm3CompilerTester_Assignment)
target_include_directories(qasm3CompilerTester_Assignment PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_Assignment qcor-mlir-api gtest gtest_main)

add_executable(qasm3CompilerTester_TestUtils test_qcor_expect_true.cpp)
add_test(NAME qcor_qasm3_test_utils_tester COMMAND qasm3CompilerTester_TestUtils)
target_include_directories(qasm3CompilerTester_TestUtils PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_TestUtils qcor-mlir-api gtest gtest_main)

