add_test(NAME qsim_vqe COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/VqeWithAnsatzCircuit.cpp)
add_test(NAME qsim_trotter COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/TrotterTdWorkflow.cpp)
