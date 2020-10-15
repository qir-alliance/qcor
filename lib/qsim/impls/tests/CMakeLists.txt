add_test(NAME qsim_vqe COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/VqeWithAnsatzCircuit.cpp)
add_test(NAME qsim_trotter COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/TrotterTdWorkflow.cpp)
add_test(NAME qsim_iqpe_simple COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/IterativeQpeSimple.cpp)
add_test(NAME qsim_iqpe_vqe COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/IterativeQpeVqe.cpp)
