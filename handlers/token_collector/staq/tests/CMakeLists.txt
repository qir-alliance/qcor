
configure_file(StaqTokenCollectorTester.in.cpp
               ${CMAKE_BINARY_DIR}/handlers/token_collector/staq/tests/StaqTokenCollectorTester.cpp)

link_directories(${XACC_ROOT}/lib)
add_executable(StaqTokenCollectorTester ${CMAKE_BINARY_DIR}/handlers/token_collector/staq/tests/StaqTokenCollectorTester.cpp)
add_test(NAME qcor_StaqTokenCollectorTester COMMAND StaqTokenCollectorTester)
target_include_directories(StaqTokenCollectorTester PRIVATE . ../../ ../../tests ${XACC_ROOT}/include/gtest)
target_link_libraries(StaqTokenCollectorTester ${XACC_TEST_LIBRARIES} xacc::xacc test_utils)
configure_file(${CMAKE_SOURCE_DIR}/examples/adder/add_3_5.v ${CMAKE_BINARY_DIR}/handlers/token_collector/staq/tests COPYONLY)
