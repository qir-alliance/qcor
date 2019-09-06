set(CMAKE_CXX_COMPILER qcor)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -a local-ibm")
message(STATUS "LINKING ${XACC_ROOT}/lib")

link_directories(${XACC_ROOT}/lib)
add_executable(test_bell_state test_bell_state.cpp)
target_include_directories(test_bell_state PRIVATE ${XACC_INCLUDE_ROOT}/gtest)
add_test(test_bell_state test_bell_state)
target_link_libraries(test_bell_state PRIVATE gtest gtest_main)
