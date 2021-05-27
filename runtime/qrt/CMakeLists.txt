set(LIBRARY_NAME qrt)

file(GLOB SRC *.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME} 
                            PUBLIC 
                            . 
                            ${CMAKE_BINARY_DIR} 
                            ${XACC_ROOT}/include/eigen
                            ${CMAKE_SOURCE_DIR}/tpls/argparse)

target_link_libraries(${LIBRARY_NAME} PUBLIC xacc::xacc xacc::quantum_gate xacc::pauli)

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

file(GLOB HEADERS qrt.hpp)
install(FILES ${HEADERS} DESTINATION include/qcor)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

add_subdirectory(impls)