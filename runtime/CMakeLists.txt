
configure_file("${CMAKE_SOURCE_DIR}/runtime/qcor_version.in"
               "${CMAKE_BINARY_DIR}/runtime/qcor_version")
install(FILES "${CMAKE_BINARY_DIR}/runtime/qcor_version" DESTINATION include/qcor)

set(LIBRARY_NAME qcor)

file(GLOB SRC observable/qcor_observable.cpp 
              optimizer/qcor_optimizer.cpp 
              objectives/objective_function.cpp
              kernel/quantum_kernel.cpp
              utils/qcor_utils.cpp
              objectives/gradient_function.cpp)

              #execution/taskInitiate.cpp
              #utils/qcor_types.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME} PUBLIC . objectives qrt qrt/internal_compiler qrt/ir ${XACC_ROOT}/include/spdlog
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

file(GLOB HEADERS qcor.hpp qcor_lang_ext.hpp
                  observable/qcor_observable.hpp 
                  optimizer/qcor_optimizer.hpp 
                  kernel/quantum_kernel.hpp 
                  qrt/qrt.hpp qrt/ir/*.hpp 
                  qrt/internal_compiler/qalloc 
                  qrt/internal_compiler/*.hpp
                  objectives/objective_function.hpp 
                  objectives/gradient_function.hpp
                  execution/taskInitiate.hpp
                  utils/qcor_utils.hpp
                  utils/qcor_pimpl.hpp
                  jit/qcor_jit.hpp)

install(FILES ${HEADERS} DESTINATION include/qcor)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
set(EXTRA_INCLUDE "")
foreach(EXTRA_HEADER ${QCOR_EXTRA_HEADERS})
  set(EXTRA_INCLUDE "${EXTRA_INCLUDE} -I ${EXTRA_HEADER}")
endforeach()

if (EXTRA_INCLUDE)
  string(REGEX REPLACE "\"" "" EXTRA_INCLUDE ${EXTRA_INCLUDE})
  message(STATUS "Extra Includes for PCH-generation: ${EXTRA_INCLUDE}")
endif()

install(CODE "execute_process(COMMAND ${LLVM_INSTALL_PREFIX}/bin/clang++ -std=c++17 -x c++-header ${CMAKE_INSTALL_PREFIX}/include/qcor/qcor_lang_ext.hpp -o ${CMAKE_BINARY_DIR}/runtime/qcor_lang_ext.hpp.pch -I ${XACC_ROOT}/include/xacc -I ${CMAKE_INSTALL_PREFIX}/include/qcor -I ${XACC_ROOT}/include/eigen ${EXTRA_INCLUDE})")
install(FILES ${CMAKE_BINARY_DIR}/runtime/qcor_lang_ext.hpp.pch DESTINATION include/qcor)

if (QCOR_BUILD_TESTS)
  add_subdirectory(tests)
  add_subdirectory(observable/tests)
  add_subdirectory(optimizer/tests)
endif()

add_subdirectory(objectives)
add_subdirectory(qrt)
add_subdirectory(jit)
