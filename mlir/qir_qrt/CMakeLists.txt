add_library(qir-qrt OBJECT qir-qrt.cpp)
#set_source_files_properties(qir-qrt.cpp PROPERTIES COMPILE_FLAGS "-emit-llvm" )

target_include_directories(
  qir-qrt
  PUBLIC . ${XACC_ROOT}/include/cppmicroservices4 ${XACC_ROOT}/include/xacc ${XACC_ROOT}/include/qcor ${XACC_ROOT}/include/eigen ${XACC_ROOT}/include/quantum/gate)

install (TARGETS qir-qrt DESTINATION lib)