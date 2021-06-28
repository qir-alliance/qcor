
configure_file(qcor_jit.in.cpp
               ${CMAKE_BINARY_DIR}/runtime/jit/qcor_jit.cpp)

set(LIBRARY_NAME qcor-jit)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
add_library(${LIBRARY_NAME}
            SHARED
            ${CMAKE_BINARY_DIR}/runtime/jit/qcor_jit.cpp)

target_include_directories(${LIBRARY_NAME}
                           PUBLIC . ../qrt ${CMAKE_SOURCE_DIR}/runtime/qrt/internal_compiler ${XACC_ROOT}/include/xacc ${CMAKE_SOURCE_DIR}/handlers
                                  ${CLANG_INCLUDE_DIRS}
                                  ${LLVM_INCLUDE_DIRS})

target_link_libraries(${LIBRARY_NAME} 
                      PRIVATE ${CLANG_LIBS} ${LLVM_LIBS} qcor-syntax-handler qcor-clang-wrapper)
if(APPLE)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "@loader_path;${CMAKE_INSTALL_PREFIX}/clang-plugins;${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "$ORIGIN:${LLVM_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/clang-plugins")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

install(FILES qcor_jit.hpp DESTINATION include/qcor)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
