configure_file(QCORASTVisitorTester.in.cpp
               ${CMAKE_BINARY_DIR}/compiler/tests/QCORASTVisitorTester.cpp)
include_directories(${CMAKE_SOURCE_DIR}/compiler)

add_xacc_test(QCORASTVisitor)
link_directories(/usr/lib/x86_64-linux-gnu)
target_link_libraries(QCORASTVisitorTester rt dl tinfo pthread m qcor qcor-ast-plugin xacc::quantum_gate ${CLANG_LIBS})

