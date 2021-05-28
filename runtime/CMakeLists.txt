
configure_file("${CMAKE_SOURCE_DIR}/runtime/qcor_version.in"
               "${CMAKE_BINARY_DIR}/runtime/qcor_version")
install(FILES "${CMAKE_BINARY_DIR}/runtime/qcor_version" DESTINATION include/qcor)

set(LIBRARY_NAME qcor)

file(GLOB SRC observable/qcor_observable.cpp 
              optimizer/qcor_optimizer.cpp 
              objectives/objective_function.cpp
              objectives/gradient_function.cpp
              execution/taskInitiate.cpp
              utils/qcor_utils.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME} PUBLIC . qrt qrt/internal_compiler
                            observable 
                            optimizer 
                            jit
                            kernel 
                            objectives 
                            execution 
                            utils)

target_link_libraries(${LIBRARY_NAME} PUBLIC xacc::xacc xacc::quantum_gate qrt xacc::pauli xacc::fermion)

if(APPLE)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "${XACC_ROOT}/lib")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "${XACC_ROOT}/lib")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

file(GLOB HEADERS qcor.hpp 
                  observable/qcor_observable.hpp 
                  optimizer/qcor_optimizer.hpp 
                  kernel/quantum_kernel.hpp 
                  objectives/objective_function.hpp 
                  objectives/gradient_function.hpp
                  execution/taskInitiate.hpp
                  utils/qcor_utils.hpp)
                  
install(FILES ${HEADERS} DESTINATION include/qcor)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

if (QCOR_BUILD_TESTS)
  add_subdirectory(tests)
endif()

add_subdirectory(objectives)
add_subdirectory(qrt)
add_subdirectory(jit)