add_executable(TimeSeriesQpeTester TimeSeriesQpeTester.cpp)
add_test(NAME qcor_TimeSeriesQpeTester COMMAND TimeSeriesQpeTester)
target_include_directories(TimeSeriesQpeTester PRIVATE ../../ ../../../base ${XACC_ROOT}/include/gtest ${XACC_ROOT}/include/eigen)
link_directories(${XACC_ROOT}/lib)
target_link_libraries(TimeSeriesQpeTester ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so xacc::xacc xacc::quantum_gate qcor-qsim)


add_executable(TimeSeriesQpeNoiseTester TimeSeriesQpeNoiseTester.cpp)
add_test(NAME qcor_TimeSeriesQpeNoiseTester COMMAND TimeSeriesQpeNoiseTester)
target_include_directories(TimeSeriesQpeNoiseTester PRIVATE ../../ ../../../base ${XACC_ROOT}/include/gtest ${XACC_ROOT}/include/eigen)
link_directories(${XACC_ROOT}/lib)
target_compile_definitions(TimeSeriesQpeNoiseTester PRIVATE RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
target_link_libraries(TimeSeriesQpeNoiseTester ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so xacc::xacc xacc::quantum_gate qcor-qsim)