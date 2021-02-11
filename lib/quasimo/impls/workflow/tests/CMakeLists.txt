link_directories(${XACC_ROOT}/lib)

if (NOT XACC_ROOT STREQUAL CMAKE_INSTALL_PREFIX)
  add_definitions(-D__internal__qcor__compile__plugin__path="${CMAKE_INSTALL_PREFIX}/plugins")
endif()

add_executable(VqeWorkflowTester VqeWorkflowTester.cpp)
add_test(NAME qcor_VqeWorkflowTester COMMAND VqeWorkflowTester)
target_include_directories(VqeWorkflowTester PRIVATE ../../ ../../../base ${XACC_ROOT}/include/gtest)
target_link_libraries(VqeWorkflowTester ${XACC_TEST_LIBRARIES} xacc::xacc xacc::quantum_gate qcor-quasimo)

add_executable(QaoaWorkflowTester QaoaWorkflowTester.cpp)
add_test(NAME QaoaWorkflowTester COMMAND QaoaWorkflowTester)
target_include_directories(QaoaWorkflowTester PRIVATE ../../ ../../../base ${XACC_ROOT}/include/gtest)
target_link_libraries(QaoaWorkflowTester ${XACC_TEST_LIBRARIES} xacc::xacc xacc::quantum_gate qcor-quasimo)

add_executable(QiteWorkflowTester QiteWorkflowTester.cpp)
add_test(NAME QiteWorkflowTester COMMAND QiteWorkflowTester)
target_include_directories(QiteWorkflowTester PRIVATE ../../ ../../../base ${XACC_ROOT}/include/gtest)
target_link_libraries(QiteWorkflowTester ${XACC_TEST_LIBRARIES} xacc::xacc xacc::quantum_gate qcor-quasimo)

add_executable(AdaptVqeWorkflowTester AdaptVqeWorkflowTester.cpp)
# Don't run this test since it's unstable
# add_test(NAME AdaptVqeWorkflowTester COMMAND AdaptVqeWorkflowTester)
target_include_directories(AdaptVqeWorkflowTester PRIVATE ../../ ../../../base ${XACC_ROOT}/include/gtest)
target_link_libraries(AdaptVqeWorkflowTester ${XACC_TEST_LIBRARIES} xacc::xacc xacc::quantum_gate qcor-quasimo)