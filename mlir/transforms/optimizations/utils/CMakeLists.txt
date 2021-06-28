set(LIBRARY_NAME MLIRQuantumUtils)

file(GLOB SRC gate_matrix.cpp)

add_library(${LIBRARY_NAME} STATIC ${SRC})

target_include_directories(${LIBRARY_NAME} PUBLIC . ${XACC_ROOT}/include/eigen)