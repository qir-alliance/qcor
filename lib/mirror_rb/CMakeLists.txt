set(LIBRARY_NAME qcor-mirror-rb)

file(GLOB SRC *.cpp)
file(GLOB HEADERS *.hpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})
target_include_directories(${LIBRARY_NAME} PUBLIC . ${XACC_ROOT}/include/eigen)
target_link_libraries(${LIBRARY_NAME} PUBLIC qcor qrt)
xacc_configure_library_rpath(${LIBRARY_NAME})

install(FILES ${HEADERS} DESTINATION include/qcor)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

if (QCOR_BUILD_TESTS)
  add_subdirectory(tests)
endif()