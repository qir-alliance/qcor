add_xacc_test(YPulse)
target_link_libraries(YPulseTester qcor-instructions)

add_xacc_test(HWE)
target_link_libraries(HWETester qcor-instructions xacc-pauli)