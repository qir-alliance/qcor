


link_directories(${XACC_ROOT}/lib)
add_executable(PyXASMTokenCollectorTester PyXASMTokenCollectorTester.cpp)
add_test(NAME qcor_PyXASMTokenCollectorTester COMMAND PyXASMTokenCollectorTester)
target_include_directories(PyXASMTokenCollectorTester PRIVATE . ../../ ../../tests ${CMAKE_BINARY_DIR} ${XACC_ROOT}/include/gtest)
target_link_libraries(PyXASMTokenCollectorTester ${XACC_TEST_LIBRARIES} xacc::xacc test_utils)

