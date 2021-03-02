link_directories(${XACC_ROOT}/lib)
if (NOT XACC_ROOT STREQUAL CMAKE_INSTALL_PREFIX)
  add_definitions(-D__internal__qcor__compile__plugin__path="${CMAKE_INSTALL_PREFIX}/plugins")
endif()

add_executable(NisqQrtTester NisqQrtTester.cpp)
add_test(NAME qcor_NisqQrtTester COMMAND NisqQrtTester)
target_include_directories(NisqQrtTester PRIVATE ${XACC_ROOT}/include/gtest)
target_link_libraries(NisqQrtTester ${XACC_TEST_LIBRARIES} qcor)