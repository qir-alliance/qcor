link_directories(${XACC_ROOT}/lib)
if (NOT XACC_ROOT STREQUAL CMAKE_INSTALL_PREFIX)
  add_definitions(-D__internal__qcor__compile__plugin__path="${CMAKE_INSTALL_PREFIX}/plugins")
endif()

add_executable(FtqcQrtTester FtqcQrtTester.cpp)
add_test(NAME qcor_FtqcQrtTester COMMAND FtqcQrtTester)
target_include_directories(FtqcQrtTester PRIVATE ${XACC_ROOT}/include/gtest)
target_link_libraries(FtqcQrtTester ${XACC_TEST_LIBRARIES} qcor)