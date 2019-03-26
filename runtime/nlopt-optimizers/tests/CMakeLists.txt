include_directories(${CMAKE_SOURCE_DIR}/runtime/nlopt-optimizers)
include_directories(${CMAKE_SOURCE_DIR}/runtime)
include_directories(${CMAKE_SOURCE_DIR}/runtime/nlopt-optimizers)
include_directories(${CMAKE_BUILD_DIR}/runtime/nlopt-optimizers/nlopt/src/api)

add_xacc_test(NLOptimizer)
target_link_libraries(NLOptimizerTester nlopt qcor-runtime-nlopt qcor)