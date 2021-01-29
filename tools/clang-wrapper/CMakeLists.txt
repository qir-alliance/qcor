set(LIBRARY_NAME qcor-clang-wrapper)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

configure_file(qcor_clang_wrapper.in.cpp
               ${CMAKE_BINARY_DIR}/tools/clang-wrapper/qcor_clang_wrapper.cpp)

add_library(${LIBRARY_NAME}
            SHARED
            ${CMAKE_BINARY_DIR}/tools/clang-wrapper/qcor_clang_wrapper.cpp)

target_include_directories(${LIBRARY_NAME}
                           PUBLIC .
                                  ${CLANG_INCLUDE_DIRS}
                                  ${LLVM_INCLUDE_DIRS})

target_link_libraries(${LIBRARY_NAME}
                      PRIVATE ${CLANG_LIBS} ${LLVM_LIBS})

if(APPLE)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "@loader_path/../lib;${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "$ORIGIN/../lib:${LLVM_INSTALL_PREFIX}/lib")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
