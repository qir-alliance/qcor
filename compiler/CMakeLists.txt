set(LIBRARY_NAME qcor-ast-plugin)
add_library(${LIBRARY_NAME}
            SHARED
            fuzzy_parsing.cpp
            qcor_ast_visitor.cpp
            qcor_ast_consumer.cpp)

target_include_directories(${LIBRARY_NAME}
                           PUBLIC .
                                  ${CLANG_INCLUDE_DIRS}
                                  ${LLVM_INCLUDE_DIRS})


target_link_libraries(${LIBRARY_NAME}
                      PUBLIC ${CLANG_LIBS} ${LLVM_LIBS} tinfo xacc::xacc)

xacc_configure_library_rpath(${LIBRARY_NAME})

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

if(QCOR_BUILD_TESTS)
  add_subdirectory(tests)
endif()

configure_file(qcor-driver.in.cpp
               ${CMAKE_BINARY_DIR}/compiler/qcor-driver.cpp)
#add_executable(qcor-driver ${CMAKE_BINARY_DIR}/compiler/qcor-driver.cpp)
#target_link_libraries(qcor-driver PRIVATE qcor-ast-plugin qcor)
#install(PROGRAMS ${CMAKE_BINARY_DIR}/compiler/qcor-driver DESTINATION bin)
