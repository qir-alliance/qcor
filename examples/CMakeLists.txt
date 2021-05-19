# Add a QCOR compilation and run test.
# The test can use qcor_expect macro to check expected results.
function(add_qcor_compile_and_exe_test test_name relative_source_location)
  add_test(
  NAME
    ${test_name}
  COMMAND
    bash -c "${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/${relative_source_location} -o ${test_name}; \
              ${CMAKE_CURRENT_BINARY_DIR}/${test_name}"
  )
endfunction()

add_test(NAME qrt_bell_multi COMMAND ${CMAKE_BINARY_DIR}/qcor -c ${CMAKE_CURRENT_SOURCE_DIR}/bell/bell_multi_qreg.cpp)
add_test(NAME qrt_add_3_5 COMMAND ${CMAKE_BINARY_DIR}/qcor -v -c ${CMAKE_CURRENT_SOURCE_DIR}/adder/add_3_5.cpp WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/adder)
add_test(NAME qrt_mixed_language COMMAND ${CMAKE_BINARY_DIR}/qcor -c ${CMAKE_CURRENT_SOURCE_DIR}/simple/mixed_language.cpp)
add_test(NAME qrt_simple-demo COMMAND ${CMAKE_BINARY_DIR}/qcor -c ${CMAKE_CURRENT_SOURCE_DIR}/simple/simple-demo.cpp)
add_test(NAME qrt_deuteron_exp_inst COMMAND ${CMAKE_BINARY_DIR}/qcor -c ${CMAKE_CURRENT_SOURCE_DIR}/deuteron/deuteron_exp_inst.cpp)
add_test(NAME qrt_deuteron_task_initiate COMMAND ${CMAKE_BINARY_DIR}/qcor -c ${CMAKE_CURRENT_SOURCE_DIR}/deuteron/deuteron_task_initiate.cpp)
add_test(NAME qrt_qaoa_example COMMAND ${CMAKE_BINARY_DIR}/qcor -c ${CMAKE_CURRENT_SOURCE_DIR}/qaoa/qaoa_example.cpp)
add_test(NAME qrt_simple-objective-function COMMAND ${CMAKE_BINARY_DIR}/qcor -c ${CMAKE_CURRENT_SOURCE_DIR}/simple/simple-objective-function.cpp)
add_test(NAME qrt_qpe_example COMMAND ${CMAKE_BINARY_DIR}/qcor -c ${CMAKE_CURRENT_SOURCE_DIR}/qpe/qpe_example_qrt.cpp)
add_test(NAME qrt_kernel_include COMMAND ${CMAKE_BINARY_DIR}/qcor -c ${CMAKE_CURRENT_SOURCE_DIR}/simple/multiple_kernels.cpp)
add_test(NAME qrt_grover COMMAND ${CMAKE_BINARY_DIR}/qcor -c ${CMAKE_CURRENT_SOURCE_DIR}/grover/grover.cpp)
add_test(NAME qrt_adapt COMMAND ${CMAKE_BINARY_DIR}/qcor -c ${CMAKE_CURRENT_SOURCE_DIR}/hybrid/adapt_h2.cpp)
add_test(NAME qrt_ftqc_simple COMMAND ${CMAKE_BINARY_DIR}/qcor -c -qrt ftqc ${CMAKE_CURRENT_SOURCE_DIR}/ftqc_qrt/simple-demo.cpp)
add_test(NAME qrt_ftqc_rus COMMAND ${CMAKE_BINARY_DIR}/qcor -c -qrt ftqc ${CMAKE_CURRENT_SOURCE_DIR}/ftqc_qrt/repeat-until-success.cpp)
add_test(NAME qrt_ftqc_qec COMMAND ${CMAKE_BINARY_DIR}/qcor -c -qrt ftqc ${CMAKE_CURRENT_SOURCE_DIR}/ftqc_qrt/bit-flip-code.cpp)
add_test(NAME qrt_ftqc_deuteron COMMAND ${CMAKE_BINARY_DIR}/qcor -c -qrt ftqc ${CMAKE_CURRENT_SOURCE_DIR}/ftqc_qrt/deuteron.cpp)
add_test(NAME qrt_ftqc_bit_flip_qec_std_lib COMMAND ${CMAKE_BINARY_DIR}/qcor -c -qrt ftqc ${CMAKE_CURRENT_SOURCE_DIR}/ftqc_qrt/error_correcting_code.cpp)
add_test(NAME qrt_ftqc_five_qubit_qec_std_lib COMMAND ${CMAKE_BINARY_DIR}/qcor -c -qrt ftqc ${CMAKE_CURRENT_SOURCE_DIR}/ftqc_qrt/five_qubit_qec_code.cpp)
add_test(NAME qrt_ftqc_steane_qec_std_lib COMMAND ${CMAKE_BINARY_DIR}/qcor -c -qrt ftqc ${CMAKE_CURRENT_SOURCE_DIR}/ftqc_qrt/steane_qec_code.cpp)
add_test(NAME quasimo_vqe COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/quasimo/VqeWithAnsatzCircuit.cpp)
add_test(NAME quasimo_trotter COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/quasimo/TrotterTdWorkflow.cpp)
add_test(NAME quasimo_iqpe_vqe COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/quasimo/IterativeQpeVqe.cpp)
add_test(NAME quasimo_adapt_vqe COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/quasimo/AdaptVqeWorkflow.cpp)
add_test(NAME quasimo_qaoa COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/quasimo/QaoaWorkflow.cpp)
add_test(NAME quasimo_qite COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/quasimo/QiteWorkflow.cpp)
add_test(NAME quasimo_heisenberg COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/quasimo/TdWorkflowHeisenbergModel.cpp)
add_test(NAME quasimo_verified_qpe COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/quasimo/VerifiedQuantumPhaseEstimation.cpp)
add_test(NAME hadamard_ctrl_test COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/ctrl-gates/simple_hadamard_test.cpp)
add_test(NAME multi_ctrl_test COMMAND ${CMAKE_BINARY_DIR}/qcor ${CMAKE_CURRENT_SOURCE_DIR}/ctrl-gates/multiple_controls.cpp)

add_qcor_compile_and_exe_test(qrt_bell_ctrl bell/bell_control.cpp)

# Lambda tests
add_qcor_compile_and_exe_test(qrt_qpu_lambda_simple qpu_lambda/lambda_test.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_lambda_bell qpu_lambda/lambda_test_bell.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_lambda_grover qpu_lambda/grover_lambda_oracle.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_lambdas_in_loop qpu_lambda/deuteron_lambda.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_lambda_deuteron qpu_lambda/deuteron_vqe.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_lambda_objfunc qpu_lambda/deuteron_vqe_obj_func.cpp)

# Arithmetic tests
add_qcor_compile_and_exe_test(qrt_qpu_arith_adder arithmetic/simple.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_arith_integer_add arithmetic/integer_add.cpp)