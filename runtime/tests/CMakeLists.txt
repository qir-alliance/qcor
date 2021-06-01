link_directories(${XACC_ROOT}/lib)
if (NOT XACC_ROOT STREQUAL CMAKE_INSTALL_PREFIX)
  add_definitions(-D__internal__qcor__compile__plugin__path="${CMAKE_INSTALL_PREFIX}/plugins")
endif()

add_executable(QCORTester QCORTester.cpp)
add_test(NAME qcor_QCORTester COMMAND QCORTester)
target_include_directories(QCORTester PRIVATE ${XACC_ROOT}/include/gtest)
target_link_libraries(QCORTester ${XACC_TEST_LIBRARIES} qcor)


configure_file(InternalCompilerTester.in.cpp
               ${CMAKE_BINARY_DIR}/runtime/tests/InternalCompilerTester.cpp)
add_executable(InternalCompilerTester ${CMAKE_BINARY_DIR}/runtime/tests/InternalCompilerTester.cpp)
target_include_directories(InternalCompilerTester PRIVATE ${XACC_ROOT}/include/gtest ${CMAKE_SOURCE_DIR}/runtime/qrt/internal_compiler)
target_link_libraries(InternalCompilerTester PRIVATE xacc qrt ${XACC_TEST_LIBRARIES})
add_test(NAME qcor_InternalCompilerTester COMMAND InternalCompilerTester)
target_compile_features(InternalCompilerTester PRIVATE cxx_std_17)


add_executable(QuilCompilerQallocTester QuilCompilerQallocTester.cpp)
add_test(NAME qcor_QuilCompilerQallocTester COMMAND QuilCompilerQallocTester)
target_include_directories(QuilCompilerQallocTester PRIVATE ${XACC_ROOT}/include/gtest ${CMAKE_SOURCE_DIR}/runtime/qrt/internal_compiler)
target_link_libraries(QuilCompilerQallocTester ${XACC_TEST_LIBRARIES} qcor qrt ${XACC_TEST_LIBRARIES})

add_executable(XASMCompilerQallocTester XASMCompilerQallocTester.cpp)
add_test(NAME qcor_XASMCompilerQallocTester COMMAND XASMCompilerQallocTester)
target_include_directories(XASMCompilerQallocTester PRIVATE ${XACC_ROOT}/include/gtest ${CMAKE_SOURCE_DIR}/runtime/qrt/internal_compiler)
target_link_libraries(XASMCompilerQallocTester ${XACC_TEST_LIBRARIES} qcor qrt ${XACC_TEST_LIBRARIES})