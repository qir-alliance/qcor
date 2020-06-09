
configure_file(test.in.sh
               ${CMAKE_BINARY_DIR}/tools/qopt/passes/xacc-ir-transformation/tests/test.sh)

add_test(NAME test_qrt_qopt_xacc_opt COMMAND sh test.sh)
