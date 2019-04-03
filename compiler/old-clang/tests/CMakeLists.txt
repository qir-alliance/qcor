include_directories(${CMAKE_SOURCE_DIR}/compiler/clang)

add_xacc_test(LambdaVisitor)
target_link_libraries(LambdaVisitorTester qcor-ast-plugin ${CLANG_LIBS})