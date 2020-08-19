set(LIBRARY_NAME qcor)

file(GLOB SRC observable/qcor_observable.cpp 
              optimizer/qcor_optimizer.cpp 
              objectives/objective_function.cpp
              execution/taskInitiate.cpp
              utils/qcor_utils.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME} PUBLIC . qrt 
                            observable 
                            optimizer 
                            kernel 
                            objectives 
                            execution 
                            utils)

target_link_libraries(${LIBRARY_NAME} PUBLIC xacc::xacc xacc::quantum_gate qrt xacc::pauli)

xacc_configure_library_rpath(${LIBRARY_NAME})

file(GLOB HEADERS qcor.hpp 
                  observable/qcor_observable.hpp 
                  optimizer/qcor_optimizer.hpp 
                  kernel/quantum_kernel.hpp 
                  objectives/objective_function.hpp 
                  execution/taskInitiate.hpp
                  #utils/eigen_qcor_unitary_addon.hpp
                  utils/qcor_utils.hpp)
                  
install(FILES ${HEADERS} DESTINATION include/qcor)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

if (QCOR_BUILD_TESTS)
  add_subdirectory(tests)
endif()

add_subdirectory(objectives)
add_subdirectory(qrt)
add_subdirectory(jit)