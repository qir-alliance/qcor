
set(LIBRARY_NAME token_collector_helper)
add_library(${LIBRARY_NAME}
            SHARED
            token_collector_helper.cpp)

target_include_directories(${LIBRARY_NAME}
                           PUBLIC . ${CMAKE_SOURCE_DIR}/handler 
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
