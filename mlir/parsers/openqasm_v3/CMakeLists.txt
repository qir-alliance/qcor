
set(LIBRARY_NAME openqasmv3-mlir-generator)

set(ANTLR_LIB ${XACC_ROOT}/lib/libantlr4-runtime.so)
if (APPLE)
  set(ANTLR_LIB ${XACC_ROOT}/lib/libantlr4-runtime.dylib)
endif()

file(GLOB SRC *.cpp generated/*.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})
target_compile_features(${LIBRARY_NAME} 
                        PUBLIC
                        cxx_std_17)
target_compile_options(${LIBRARY_NAME} PUBLIC "-Wno-attributes -Wno-suggest-override")
target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC . .. ../../dialect/include ${CMAKE_BINARY_DIR}/mlir/dialect/include ${XACC_ROOT}/include/antlr4-runtime generated)

target_link_libraries(${LIBRARY_NAME} PUBLIC MLIRQuantum MLIRIR MLIRStandard MLIRAffine ${ANTLR_LIB})


if(APPLE)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${MLIR_INSTALL_DIR}/lib;${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${MLIR_INSTALL_DIR}/lib:${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

add_subdirectory(tests)