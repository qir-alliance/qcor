macro(qcor_enable_rpath LIBNAME)
  if(APPLE)
    set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH "@loader_path/../lib")
    set_target_properties(${LIBNAME}
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  else()
    set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
    set_target_properties(${LIBNAME} PROPERTIES LINK_FLAGS "-shared")
  endif()
endmacro()

set(LIBRARY_NAME qcor)

file(GLOB SRC *.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME} PUBLIC . algorithm ${XACC_INCLUDE_DIRS})

target_link_libraries(${LIBRARY_NAME} PUBLIC xacc xacc-pauli)

if(QCOR_BUILD_TESTS)
  # add_subdirectory(tests)
endif()

if(APPLE)
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "@loader_path")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

file(GLOB HEADERS *.hpp)
install(FILES ${HEADERS} DESTINATION include/qcor)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

add_subdirectory(algorithms)
add_subdirectory(nlopt-optimizers)