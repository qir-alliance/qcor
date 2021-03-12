add_llvm_executable(qcor-mlir-tool qcor-mlir-tool.cpp)

target_compile_options(qcor-mlir-tool PUBLIC "-fexceptions")

target_compile_features(qcor-mlir-tool
                        PUBLIC
                        cxx_std_17)
llvm_update_compile_flags(qcor-mlir-tool)
target_link_libraries(qcor-mlir-tool PUBLIC MLIRQuantum quantum-to-llvm-lowering openqasm-mlir-generator openqasmv3-mlir-generator)

if(APPLE)
  set_target_properties(qcor-mlir-tool
                        PROPERTIES INSTALL_RPATH "${MLIR_INSTALL_DIR}/lib;${CMAKE_INSTALL_PREFIX}/lib")
else()
set_target_properties(qcor-mlir-tool
                        PROPERTIES INSTALL_RPATH "${MLIR_INSTALL_DIR}/lib:${CMAKE_INSTALL_PREFIX}/lib")
endif()

#set_target_properties(qcor-mlir-tool
#                        PROPERTIES INSTALL_RPATH "${MLIR_INSTALL_DIR}/lib:${CMAKE_BINARY_DIR}/mlir/parsers/openqasm:${CMAKE_BINARY_DIR}/lib")
install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/qcor-mlir-tool DESTINATION bin)
