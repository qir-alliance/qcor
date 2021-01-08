add_llvm_executable(qcor-qasm qcor-qasm.cpp)

target_compile_options(qcor-qasm PUBLIC "-fexceptions")

target_compile_features(qcor-qasm
                        PUBLIC
                        cxx_std_17)
llvm_update_compile_flags(qcor-qasm)
target_link_libraries(qcor-qasm PUBLIC quantum-to-llvm-lowering openqasm-mlir-generator )

set_target_properties(qcor-qasm
                        PROPERTIES INSTALL_RPATH "${MLIR_INSTALL_DIR}/lib:${CMAKE_BINARY_DIR}/mlir/parsers/openqasm:${CMAKE_BINARY_DIR}/lib")
install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/qcor-qasm DESTINATION bin)
