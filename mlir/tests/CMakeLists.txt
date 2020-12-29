link_directories(${XACC_ROOT}/lib)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
message(STATUS "DIALECT LIBS: ${dialect_libs}")
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        quantum-dialect
        MLIRTargetLLVMIR
        MLIRExecutionEngine
        )

add_llvm_executable(QasmTester QasmTester.cpp)

target_compile_options(QasmTester PUBLIC "-fexceptions")

target_compile_features(QasmTester
                        PUBLIC
                        cxx_std_17)
target_include_directories(QasmTester PRIVATE . ../dialect)

llvm_update_compile_flags(QasmTester)
target_link_libraries(QasmTester PUBLIC ${LIBS} staq-mlir-visitor)

set_target_properties(QasmTester
                        PROPERTIES INSTALL_RPATH "${CMAKE_BINARY_DIR}/mlir/parsers/openqasm")
mlir_check_all_link_libraries(QasmTester)
