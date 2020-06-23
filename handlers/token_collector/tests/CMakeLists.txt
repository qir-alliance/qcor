
set(LIBRARY_NAME test_utils)
add_library(${LIBRARY_NAME}
            SHARED
            test_utils.cpp)

target_include_directories(${LIBRARY_NAME}
                           PUBLIC 
                                  ${CLANG_INCLUDE_DIRS}
                                  ${LLVM_INCLUDE_DIRS})
target_compile_options(${LIBRARY_NAME} PRIVATE "-fno-rtti")

target_link_libraries(${LIBRARY_NAME}
                      PUBLIC ${CLANG_LIBS} ${LLVM_LIBS})

link_directories(${XACC_ROOT}/lib)
add_executable(TokenCollectorTester TokenCollectorTester.cpp)
add_test(NAME qcor_TokenCollectorTester COMMAND TokenCollectorTester)
target_include_directories(TokenCollectorTester PRIVATE . .. ../../ ${XACC_ROOT}/include/gtest)
target_link_libraries(TokenCollectorTester ${XACC_TEST_LIBRARIES} xacc::xacc token_collector_util ${LIBRARY_NAME})

