configure_file(LambdaVisitorTester.in.cpp
               ${CMAKE_BINARY_DIR}/compiler/clang/tests/LambdaVisitorTester.cpp)
include_directories(${CMAKE_SOURCE_DIR}/compiler/clang)

add_xacc_test(LambdaVisitor)
target_link_libraries(LambdaVisitorTester qcor qcor-ast-plugin xacc::quantum_gate ${CLANG_LIBS})