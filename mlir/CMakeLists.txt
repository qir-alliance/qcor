set(LLVM_ROOT ${MLIR_DIR})
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})

get_filename_component(MLIR_INSTALL_DIR "${MLIR_DIR}/../../.." ABSOLUTE)

add_subdirectory(dialect)
add_subdirectory(parsers)
add_subdirectory(transforms)
add_subdirectory(qir_qrt)
add_subdirectory(tools)

set(LIBRARY_NAME qcor-mlir-api)

file(GLOB SRC qcor_mlir_api.cpp)

include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/runtime/jit)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRQuantum
        MLIROptLib
        MLIRTargetLLVMIR
        MLIRExecutionEngine
        MLIRStandard
        MLIRAffine
        LLVMLinker
        openqasm-mlir-generator
        openqasmv3-mlir-generator
        quantum-to-llvm-lowering
        qcor-jit
        )

add_mlir_library(${LIBRARY_NAME} SHARED ${SRC} LINK_LIBS PUBLIC ${LIBS})
target_compile_options(${LIBRARY_NAME} PUBLIC "-fexceptions")
target_compile_features(${LIBRARY_NAME} 
                        PUBLIC
                        cxx_std_17)

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC . dialect/include ${CMAKE_BINARY_DIR}/mlir/dialect/include) 

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

