set(LIBRARY_NAME qcor-ast-plugin)
add_library(${LIBRARY_NAME}
            SHARED
            QCORPluginAction.cpp
            FuzzyParsingExternalSemaSource.cpp
            LambdaVisitor.cpp
            QCORASTConsumer.cpp)
target_include_directories(${LIBRARY_NAME}
                           PUBLIC .
                                  ${CMAKE_SOURCE_DIR}/runtime
                                  ${CLANG_INCLUDE_DIRS}
                                  ${LLVM_INCLUDE_DIRS}
                                  ${XACC_INCLUDE_ROOT}
                                  ${XACC_INCLUDE_ROOT}/xacc
                                  ${XACC_INCLUDE_ROOT}/cppmicroservices4)
target_link_libraries(${LIBRARY_NAME}
                      PUBLIC ${CLANG_LIBS} ${XACC_LIBRARIES} qcor)
