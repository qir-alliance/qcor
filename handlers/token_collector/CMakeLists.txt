
set(LIBRARY_NAME token_collector_util)
add_library(${LIBRARY_NAME}
            SHARED
            token_collector_util.cpp)

target_include_directories(${LIBRARY_NAME}
                           PUBLIC . ${CMAKE_BINARY_DIR}
                                  ${CLANG_INCLUDE_DIRS}
                                  ${LLVM_INCLUDE_DIRS})

target_link_libraries(${LIBRARY_NAME}
                      PRIVATE ${CLANG_LIBS} ${LLVM_LIBS} xacc::xacc xacc::quantum_gate qrt)

if(APPLE)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "${XACC_ROOT}/lib;@loader_path")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "${XACC_ROOT}/lib:$ORIGIN")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

add_subdirectory(xasm)
add_subdirectory(staq)
#add_subdirectory(quil)
add_subdirectory(unitary)

if (QCOR_BUILD_TESTS)
  add_subdirectory(tests)
endif()