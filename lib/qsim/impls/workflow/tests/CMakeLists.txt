link_directories(${XACC_ROOT}/lib)

add_executable(VqeWorkflowTester VqeWorkflowTester.cpp)
add_test(NAME qcor_VqeWorkflowTester COMMAND VqeWorkflowTester)
target_include_directories(VqeWorkflowTester PRIVATE ../../ ../../../base ${XACC_ROOT}/include/gtest)
target_link_libraries(VqeWorkflowTester ${XACC_TEST_LIBRARIES} xacc::xacc xacc::quantum_gate qcor-qsim)

add_executable(QaoaWorkflowTester QaoaWorkflowTester.cpp)
add_test(NAME QaoaWorkflowTester COMMAND QaoaWorkflowTester)
target_include_directories(QaoaWorkflowTester PRIVATE ../../ ../../../base ${XACC_ROOT}/include/gtest)
target_link_libraries(QaoaWorkflowTester ${XACC_TEST_LIBRARIES} xacc::xacc xacc::quantum_gate qcor-qsim)

add_executable(QiteWorkflowTester QiteWorkflowTester.cpp)
add_test(NAME QiteWorkflowTester COMMAND QiteWorkflowTester)
target_include_directories(QiteWorkflowTester PRIVATE ../../ ../../../base ${XACC_ROOT}/include/gtest)
target_link_libraries(QiteWorkflowTester ${XACC_TEST_LIBRARIES} xacc::xacc xacc::quantum_gate qcor-qsim)