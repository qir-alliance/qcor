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
                      PUBLIC ${CLANG_LIBS} ${LLVM_LIBS} ${XACC_LIBRARIES} qcor)
if(APPLE)
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "@loader_path")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

if(QCOR_BUILD_TESTS)
  add_subdirectory(tests)
endif()