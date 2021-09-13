link_directories(${XACC_ROOT}/lib)

if (NOT XACC_ROOT STREQUAL CMAKE_INSTALL_PREFIX)
  add_definitions(-D__internal__qcor__compile__plugin__path="${CMAKE_INSTALL_PREFIX}/plugins")
endif()

add_executable(MirrorCircuitTester MirrorCircuitTester.cpp)
add_test(NAME qcor_MirrorCircuitTester COMMAND MirrorCircuitTester)
target_include_directories(MirrorCircuitTester PRIVATE ../../ ../../../base ${XACC_ROOT}/include/gtest)
target_link_libraries(MirrorCircuitTester ${XACC_TEST_LIBRARIES} xacc::xacc xacc::quantum_gate)

