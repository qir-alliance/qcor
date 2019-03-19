set(LIBRARY_NAME qcor-ast-plugin)
add_library(${LIBRARY_NAME} SHARED QCORPluginAction.cpp FuzzyParsingExternalSemaSource.cpp LambdaVisitor.cpp QCORASTConsumer.cpp)
target_include_directories(${LIBRARY_NAME} PUBLIC .)
target_include_directories(${LIBRARY_NAME} PUBLIC ${CLANG_INCLUDE_DIRS})
target_include_directories(${LIBRARY_NAME} PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(${LIBRARY_NAME} PUBLIC ${XACC_INCLUDE_DIRS})
target_link_libraries(${LIBRARY_NAME} PUBLIC ${CLANG_LIBS} ${XACC_LIBRARIES})

