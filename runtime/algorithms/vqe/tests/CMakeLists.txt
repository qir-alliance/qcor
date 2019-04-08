include_directories(${CMAKE_SOURCE_DIR}/runtime/algorithms/vqe)
include_directories(${CMAKE_SOURCE_DIR}/runtime)

add_xacc_test(VQE)
target_link_libraries(VQETester qcor-algorithm-vqe qcor)