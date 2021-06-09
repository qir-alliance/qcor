function(add_qcor_qsharp_test test_name relative_qs_source_location relative_cpp_source_location)
  add_test(
  NAME
    ${test_name}
  COMMAND
    bash -c "${CMAKE_BINARY_DIR}/qcor -qrt ftqc ${CMAKE_CURRENT_SOURCE_DIR}/${relative_qs_source_location} ${CMAKE_CURRENT_SOURCE_DIR}/${relative_cpp_source_location} -o ${test_name}; \
              ${CMAKE_CURRENT_BINARY_DIR}/${test_name}"
  )
endfunction()

add_qcor_qsharp_test(qcor_qsharp_bell FTQC/Bell/bell.qs FTQC/Bell/bell_driver.cpp)
add_qcor_qsharp_test(qcor_qsharp_deuteron FTQC/Deuteron/vqe_ansatz.qs FTQC/Deuteron/vqe_driver.cpp)
add_qcor_qsharp_test(qcor_qsharp_functor FTQC/Functor/functor.qs FTQC/Functor/driver.cpp)