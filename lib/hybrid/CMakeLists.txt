set(LIBRARY_NAME qcor-hybrid)

file(GLOB SRC *.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_link_libraries(${LIBRARY_NAME} PUBLIC qcor)

xacc_configure_library_rpath(${LIBRARY_NAME})

file(GLOB HEADERS qcor_hybrid.hpp)
install(FILES ${HEADERS} DESTINATION include/qcor)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

if (QCOR_BUILD_TESTS)
  add_subdirectory(tests)
endif()
