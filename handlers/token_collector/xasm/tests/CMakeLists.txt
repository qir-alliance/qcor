


link_directories(${XACC_ROOT}/lib)
add_executable(XASMTokenCollectorTester XASMTokenCollectorTester.cpp)
add_test(NAME qcor_XASMTokenCollectorTester COMMAND XASMTokenCollectorTester)
target_include_directories(XASMTokenCollectorTester PRIVATE . ../../ ../../tests ${XACC_ROOT}/include/gtest)
target_link_libraries(XASMTokenCollectorTester ${XACC_TEST_LIBRARIES} xacc::xacc test_utils)

