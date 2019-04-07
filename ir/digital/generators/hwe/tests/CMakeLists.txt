include_directories(${CMAKE_SOURCE_DIR}/ir/digital/generators/hwe)

add_xacc_test(HWE)
target_link_libraries(HWETester qcor-digital qcor)