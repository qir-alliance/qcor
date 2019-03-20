set(LIBRARY_NAME qcor)

file(GLOB SRC *.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME}
                            PUBLIC .
                                ${XACC_INCLUDE_DIRS}
                                )

target_link_libraries(${LIBRARY_NAME} PUBLIC xacc)


#qcor_enable_rpath(${LIBRARY_NAME})

if(QCOR_BUILD_TESTS)
  #add_subdirectory(tests)
endif()

file(GLOB HEADERS *.hpp)

#install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)
