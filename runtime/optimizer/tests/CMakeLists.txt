link_directories(${XACC_ROOT}/lib)
if (NOT XACC_ROOT STREQUAL CMAKE_INSTALL_PREFIX)
  add_definitions(-D__internal__qcor__compile__plugin__path="${CMAKE_INSTALL_PREFIX}/plugins")
endif()

add_executable(OptimizerTester OptimizerTester.cpp)
add_test(NAME qcor_OptimizerTester COMMAND QCORTester)
target_include_directories(OptimizerTester PRIVATE ${XACC_ROOT}/include/gtest)
target_link_libraries(OptimizerTester ${XACC_TEST_LIBRARIES} qcor)

