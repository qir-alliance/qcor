
set(LIBRARY_NAME quantum-to-llvm-lowering)

file(GLOB SRC *.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        quantum-dialect
        MLIROptLib
        MLIRTargetLLVMIR
        )

add_mlir_library(${LIBRARY_NAME} SHARED ${SRC} LINK_LIBS PUBLIC ${LIBS})
target_compile_features(${LIBRARY_NAME} 
                        PUBLIC
                        cxx_std_17)
target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC . ../dialect)

#target_link_libraries(${LIBRARY_NAME} PUBLIC ${LIBS})
