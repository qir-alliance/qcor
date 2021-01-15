add_library(qir-qrt SHARED qir-qrt.cpp)

#target_include_directories(
#  qir-qrt
#  PUBLIC . ${CMAKE_SOURCE_DIR}/runtime/qrt ${XACC_ROOT}/include/cppmicroservices4 ${XACC_ROOT}/include/xacc ${XACC_ROOT}/include/qcor ${XACC_ROOT}/include/eigen ${XACC_ROOT}/include/quantum/gate)
target_link_libraries(qir-qrt PRIVATE xacc::xacc qcor qrt)
install (TARGETS qir-qrt DESTINATION lib)