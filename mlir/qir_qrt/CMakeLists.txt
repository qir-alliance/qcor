file(GLOB SRC *.cpp)
file(GLOB HEADERS qir-types.hpp qir-qrt.hpp qir-types-utils.hpp)

set(LIBRARY_NAME qir-qrt)

add_library(qir-qrt SHARED ${SRC})

#target_include_directories(
#  qir-qrt
#  PUBLIC . ${CMAKE_SOURCE_DIR}/runtime/qrt ${XACC_ROOT}/include/cppmicroservices4 ${XACC_ROOT}/include/xacc ${XACC_ROOT}/include/qcor ${XACC_ROOT}/include/eigen ${XACC_ROOT}/include/quantum/gate)
target_link_libraries(qir-qrt PRIVATE xacc::xacc qcor qrt)

if(APPLE)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "@loader_path;${CMAKE_INSTALL_PREFIX}/lib;${XACC_ROOT}/lib")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${XACC_ROOT}/lib")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

install (TARGETS qir-qrt DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/qcor)

if (QCOR_BUILD_TESTS)
  add_subdirectory(tests)
endif()
