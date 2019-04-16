
set(LIBRARY_NAME libint-wrapper)

file(GLOB SRC *.cpp)
file(GLOB HEADERS *.hpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME} PUBLIC . ${LIBINT_INCLUDE_DIR} ${XACC_INCLUDE_DIR}/eigen)

target_link_libraries(${LIBRARY_NAME} PRIVATE ${LIBINT_LIBRARIES})

if(LIBINT_BUILD_TESTS)
  add_subdirectory(tests)
endif()


install(FILES ${HEADERS} DESTINATION include)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
