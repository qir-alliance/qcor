include_directories(${CMAKE_SOURCE_DIR}/ir/digital/generators/exp)

add_xacc_test(Exp)
target_link_libraries(ExpTester qcor-digital qcor)