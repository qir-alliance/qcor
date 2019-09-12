set(CMAKE_CXX_COMPILER ${CMAKE_BINARY_DIR}/qcor)
set(ORIGINAL_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
link_directories(${XACC_ROOT}/lib)

set(CMAKE_CXX_FLAGS "--qcor-verbose --qcor-driver-path ${CMAKE_BINARY_DIR}/compiler ${ORIGINAL_CXX_FLAGS} -a local-ibm -I${CMAKE_SOURCE_DIR}/runtime -L${CMAKE_BINARY_DIR}/runtime ")

add_executable(test_simple_kernel test_simple_kernel.cpp)
target_include_directories(test_simple_kernel PRIVATE ${XACC_INCLUDE_ROOT}/gtest)
add_test(test_simple_kernel test_simple_kernel)
target_link_libraries(test_simple_kernel PRIVATE gtest gtest_main)

add_dependencies(test_simple_kernel qcor-driver)
