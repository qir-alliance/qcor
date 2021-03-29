

set(ANTLR_LIB ${XACC_ROOT}/lib/libantlr4-runtime.so)
if (APPLE)
  set(ANTLR_LIB ${XACC_ROOT}/lib/libantlr4-runtime.dylib)
endif()

set(LIBRARY_NAME qasm3-syntax-handler)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
add_library(${LIBRARY_NAME}
            SHARED
            qasm3_handler.cpp)

target_include_directories(${LIBRARY_NAME} PUBLIC . .. ${CMAKE_SOURCE_DIR}/mlir/parsers/qasm3/antlr/generated/ ${CMAKE_SOURCE_DIR}/mlir/parsers/qasm3/utils/ ${XACC_ROOT}/include/antlr4-runtime ${CMAKE_SOURCE_DIR}/mlir/dialect/include ${CMAKE_BINARY_DIR}/mlir/dialect/include ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} )

target_link_libraries(${LIBRARY_NAME}
                      PRIVATE ${CLANG_LIBS} ${LLVM_LIBS} MLIRQuantum MLIRIR MLIRStandard qcor-mlir-api)

if(APPLE)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${XACC_ROOT}/lib;${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${XACC_ROOT}/lib:${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()


install(TARGETS ${LIBRARY_NAME} DESTINATION clang-plugins)
