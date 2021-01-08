
set(LIBRARY_NAME openqasm-mlir-generator)

file(GLOB SRC *.cpp generated/*.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})
target_compile_features(${LIBRARY_NAME} 
                        PUBLIC
                        cxx_std_17)
target_compile_options(${LIBRARY_NAME} PUBLIC "-Wno-attributes -Wno-suggest-override")
target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC . .. ../../dialect ${XACC_ROOT}/include/staq/include ${XACC_ROOT}/include/staq/libs)

target_link_libraries(${LIBRARY_NAME} PUBLIC quantum-dialect MLIRVector MLIRStandard)
