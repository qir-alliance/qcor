


link_directories(${XACC_ROOT}/lib)
#add_executable(qasm3VisitorTester test_quantum_declaration.cpp)
#add_test(NAME qcor_qasm3_quantum_decl_tester COMMAND qasm3VisitorTester)
#target_include_directories(qasm3VisitorTester PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
#target_link_libraries(qasm3VisitorTester qcor-mlir-api gtest gtest_main)

add_executable(qasm3VisitorAliasTester test_alias_handler.cpp)
add_test(NAME qcor_qasm3_quantum_alias_decl_tester COMMAND qasm3VisitorAliasTester)
target_include_directories(qasm3VisitorAliasTester PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3VisitorAliasTester qcor-mlir-api gtest gtest_main)

add_executable(qasm3CompilerTester_Assignment test_assignment.cpp)
add_test(NAME qcor_qasm3_classical_assignment_tester COMMAND qasm3CompilerTester_Assignment)
target_include_directories(qasm3CompilerTester_Assignment PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_Assignment qcor-mlir-api gtest gtest_main)

add_executable(qasm3CompilerTester_TestUtils test_qcor_expect_true.cpp)
add_test(NAME qcor_qasm3_test_utils_tester COMMAND qasm3CompilerTester_TestUtils)
target_include_directories(qasm3CompilerTester_TestUtils PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_TestUtils qcor-mlir-api gtest gtest_main)

add_executable(qasm3CompilerTester_Declaration test_declaration.cpp)
add_test(NAME qcor_qasm3_test_declaration COMMAND qasm3CompilerTester_Declaration)
target_include_directories(qasm3CompilerTester_Declaration PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_Declaration qcor-mlir-api gtest gtest_main)

add_executable(qasm3CompilerTester_Casting test_casting.cpp)
add_test(NAME qcor_qasm3_test_casting COMMAND qasm3CompilerTester_Casting)
target_include_directories(qasm3CompilerTester_Casting PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_Casting qcor-mlir-api gtest gtest_main)

add_executable(qasm3CompilerTester_LoopStmts test_loop_stmts.cpp)
add_test(NAME qcor_qasm3_test_loops COMMAND qasm3CompilerTester_LoopStmts)
target_include_directories(qasm3CompilerTester_LoopStmts PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_LoopStmts qcor-mlir-api gtest gtest_main)

add_executable(qasm3CompilerTester_QuantumInstructions test_quantum_instructions.cpp)
add_test(NAME qcor_qasm3_test_qinsts COMMAND qasm3CompilerTester_QuantumInstructions)
target_include_directories(qasm3CompilerTester_QuantumInstructions PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_QuantumInstructions qcor-mlir-api gtest gtest_main)

add_executable(qasm3CompilerTester_Subroutine test_subroutine.cpp)
add_test(NAME qcor_qasm3_test_subroutine COMMAND qasm3CompilerTester_Subroutine)
target_include_directories(qasm3CompilerTester_Subroutine PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_Subroutine qcor-mlir-api gtest gtest_main)


add_executable(qasm3CompilerTester_GlobalConstInSubroutine test_use_global_const_in_subroutine.cpp)
add_test(NAME qcor_qasm3_test_global_const_subroutine COMMAND qasm3CompilerTester_GlobalConstInSubroutine)
target_include_directories(qasm3CompilerTester_GlobalConstInSubroutine PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_GlobalConstInSubroutine qcor-mlir-api gtest gtest_main)


add_executable(qasm3CompilerTester_Arithmetic test_complex_arithmetic.cpp)
add_test(NAME qcor_qasm3_test_arithmetic COMMAND qasm3CompilerTester_Arithmetic)
target_include_directories(qasm3CompilerTester_Arithmetic PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_Arithmetic qcor-mlir-api gtest gtest_main)


add_executable(qasm3CompilerTester_ControlDirectives test_control_directives.cpp)
add_test(NAME qcor_qasm3_test_control_directives COMMAND qasm3CompilerTester_ControlDirectives)
target_include_directories(qasm3CompilerTester_ControlDirectives PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_ControlDirectives qcor-mlir-api gtest gtest_main)

add_executable(qasm3CompilerTester_Superposition test_superposition.cpp)
add_test(NAME qcor_qasm3_test_superposition COMMAND qasm3CompilerTester_Superposition)
target_include_directories(qasm3CompilerTester_Superposition PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_Superposition qcor-mlir-api gtest gtest_main)


add_executable(qasm3CompilerTester_DefaultTypes test_default_designators.cpp)
add_test(NAME qcor_qasm3_test_default_types COMMAND qasm3CompilerTester_DefaultTypes)
target_include_directories(qasm3CompilerTester_DefaultTypes PRIVATE . ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(qasm3CompilerTester_DefaultTypes qcor-mlir-api gtest gtest_main)
