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

# Compile with FTQC runtime and execute
function(add_qcor_ftqc_compile_and_exe_test test_name relative_source_location)
  add_test(
  NAME
    ${test_name}
  COMMAND
    bash -c "${CMAKE_BINARY_DIR}/qcor -qrt ftqc ${CMAKE_CURRENT_SOURCE_DIR}/${relative_source_location} -o ${test_name}; \
              ${CMAKE_CURRENT_BINARY_DIR}/${test_name}"
  )
endfunction()

add_qcor_compile_and_exe_test(qrt_bell_multi bell/bell_multi_qreg.cpp)
add_test(NAME qrt_add_3_5 COMMAND ${CMAKE_BINARY_DIR}/qcor -v -c ${CMAKE_CURRENT_SOURCE_DIR}/adder/add_3_5.cpp WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/adder)
add_qcor_compile_and_exe_test(qrt_mixed_language simple/mixed_language.cpp)
add_qcor_compile_and_exe_test(qrt_simple-demo simple/simple-demo.cpp)
add_qcor_compile_and_exe_test(qrt_deuteron_exp_inst deuteron/deuteron_exp_inst.cpp)
add_qcor_compile_and_exe_test(qrt_deuteron_task_initiate deuteron/deuteron_task_initiate.cpp)
add_qcor_compile_and_exe_test(qrt_qaoa_example qaoa/qaoa_example.cpp)
add_qcor_compile_and_exe_test(qrt_qpe_example qpe/qpe_example_qrt.cpp)
add_qcor_compile_and_exe_test(qrt_kernel_include simple/multiple_kernels.cpp)
add_qcor_compile_and_exe_test(qrt_grover grover/grover.cpp)
add_qcor_compile_and_exe_test(qrt_adapt hybrid/adapt_h2.cpp)
#add_qcor_compile_and_exe_test(compute_action_uncompute0 compute_action_uncompute/compute_action_uncompute_h2_ucc1_example.cpp)
add_qcor_compile_and_exe_test(compute_action_uncompute1 compute_action_uncompute/multiple_in_kernel.cpp)
add_qcor_compile_and_exe_test(compute_action_uncompute2 compute_action_uncompute/cau_capture_vars.cpp)

add_qcor_compile_and_exe_test(qrt_simple-explicit-parser argparse/simple-explicit-parser.cpp)
add_qcor_compile_and_exe_test(qrt_simple-implicit-parser argparse/simple-implicit-parser.cpp)

add_qcor_ftqc_compile_and_exe_test(qrt_ftqc_simple ftqc_qrt/simple-demo.cpp)
add_qcor_ftqc_compile_and_exe_test(qrt_ftqc_rus ftqc_qrt/repeat-until-success.cpp)
add_qcor_ftqc_compile_and_exe_test(qrt_ftqc_qec ftqc_qrt/bit-flip-code.cpp)
add_qcor_ftqc_compile_and_exe_test(qrt_ftqc_deuteron ftqc_qrt/deuteron.cpp)
add_qcor_ftqc_compile_and_exe_test(qrt_ftqc_bit_flip_qec_std_lib ftqc_qrt/error_correcting_code.cpp)
add_qcor_ftqc_compile_and_exe_test(qrt_ftqc_five_qubit_qec_std_lib ftqc_qrt/five_qubit_qec_code.cpp)
add_qcor_ftqc_compile_and_exe_test(qrt_ftqc_steane_qec_std_lib ftqc_qrt/steane_qec_code.cpp)
add_qcor_ftqc_compile_and_exe_test(qrt_ftqc_qalloc ftqc_qrt/qalloc_ftqc.cpp)
add_qcor_ftqc_compile_and_exe_test(qrt_ftqc_arithmetic ftqc_qrt/integer_add.cpp)
add_qcor_ftqc_compile_and_exe_test(qrt_ftqc_grover ftqc_qrt/grover.cpp)
add_qcor_compile_and_exe_test(quasimo_vqe quasimo/VqeWithAnsatzCircuit.cpp)
add_qcor_compile_and_exe_test(quasimo_trotter quasimo/TrotterTdWorkflow.cpp)
add_qcor_compile_and_exe_test(quasimo_iqpe_vqe quasimo/IterativeQpeVqe.cpp)
add_qcor_compile_and_exe_test(quasimo_adapt_vqe quasimo/AdaptVqeWorkflow.cpp)
#add_qcor_compile_and_exe_test(quasimo_qaoa quasimo/QaoaWorkflow.cpp)
add_qcor_compile_and_exe_test(quasimo_qite quasimo/QiteWorkflow.cpp)
add_qcor_compile_and_exe_test(quasimo_heisenberg quasimo/TdWorkflowHeisenbergModel.cpp)
#add_qcor_compile_and_exe_test(quasimo_verified_qpe quasimo/VerifiedQuantumPhaseEstimation.cpp)
add_qcor_compile_and_exe_test(hadamard_ctrl_test ctrl-gates/simple_hadamard_test.cpp)
add_qcor_compile_and_exe_test(multi_ctrl_test ctrl-gates/multiple_controls.cpp)

add_qcor_compile_and_exe_test(qrt_obj_func_simple simple/simple-objective-function.cpp)
add_qcor_compile_and_exe_test(qrt_kernel_autograd_simple simple/gradients_optimization.cpp)
add_qcor_compile_and_exe_test(qrt_hybrid_vqe_exe hybrid/deuteron_h2_vqe.cpp)
add_qcor_compile_and_exe_test(qrt_bell_ctrl bell/bell_control.cpp)

# Lambda tests
add_qcor_compile_and_exe_test(qrt_qpu_lambda_simple qpu_lambda/lambda_test.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_lambda_bell qpu_lambda/lambda_test_bell.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_lambda_grover qpu_lambda/grover_lambda_oracle.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_lambdas_in_loop qpu_lambda/deuteron_lambda.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_lambda_deuteron qpu_lambda/deuteron_vqe.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_lambda_objfunc qpu_lambda/deuteron_vqe_obj_func.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_lambda_hadamard_test ctrl-gates/hadamard_test.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_lambda_compute_action qpu_lambda/lambda_with_compute_action.cpp)

# Arithmetic tests
add_qcor_compile_and_exe_test(qrt_qpu_arith_adder arithmetic/simple.cpp)
add_qcor_compile_and_exe_test(qrt_qpu_arith_integer_add arithmetic/integer_add.cpp)


if (QCOR_BUILD_QSHARP_TESTS)
  add_subdirectory(qsharp)
endif()