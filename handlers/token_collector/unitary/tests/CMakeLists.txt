
link_directories(${XACC_ROOT}/lib)
add_executable(UnitaryTokenCollectorTester UnitaryTokenCollectorTester.cpp)
add_test(NAME qcor_UnitaryTokenCollectorTester COMMAND UnitaryTokenCollectorTester)
target_include_directories(UnitaryTokenCollectorTester PRIVATE . ../../ ../../tests ${CMAKE_BINARY_DIR} ${XACC_ROOT}/include/gtest)
target_link_libraries(UnitaryTokenCollectorTester ${XACC_TEST_LIBRARIES} xacc::xacc test_utils)
