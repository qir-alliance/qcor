link_directories(${XACC_ROOT}/lib)

add_executable(TimeSeriesQpeTester TimeSeriesQpeTester.cpp)
add_test(NAME qcor_TimeSeriesQpeTester COMMAND TimeSeriesQpeTester)
target_include_directories(TimeSeriesQpeTester PRIVATE ../../ ../../../base ${XACC_ROOT}/include/gtest ${XACC_ROOT}/include/eigen)
target_link_libraries(TimeSeriesQpeTester ${XACC_TEST_LIBRARIES} xacc::xacc xacc::quantum_gate qcor-qsim qcor-quantum-simulation)


add_executable(TimeSeriesQpeNoiseTester TimeSeriesQpeNoiseTester.cpp)
add_test(NAME qcor_TimeSeriesQpeNoiseTester COMMAND TimeSeriesQpeNoiseTester)
target_include_directories(TimeSeriesQpeNoiseTester PRIVATE ../../ ../../../base ${XACC_ROOT}/include/gtest ${XACC_ROOT}/include/eigen)
target_compile_definitions(TimeSeriesQpeNoiseTester PRIVATE RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
target_link_libraries(TimeSeriesQpeNoiseTester ${XACC_TEST_LIBRARIES} xacc::xacc xacc::quantum_gate qcor-qsim)

add_executable(VqeWorkflowTester VqeWorkflowTester.cpp)
add_test(NAME qcor_VqeWorkflowTester COMMAND VqeWorkflowTester)
target_include_directories(VqeWorkflowTester PRIVATE ../../ ../../../base ${XACC_ROOT}/include/gtest)
target_link_libraries(VqeWorkflowTester ${XACC_TEST_LIBRARIES} xacc::xacc xacc::quantum_gate qcor-qsim)