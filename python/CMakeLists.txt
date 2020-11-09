set(LIBRARY_NAME _pyqcor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-attributes")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -O2 -g -pipe -Werror=format-security -Wp,-D_FORTIFY_SOURCE=2 -Wformat -fexceptions --param=ssp-buffer-size=4 -grecord-gcc-switches -m64 -mtune=native -D_GNU_SOURCE -fPIC -fwrapv")
if(APPLE)
   set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif(APPLE)
file(GLOB SRC *.cpp)

set(QCOR_APPEND_PLUGIN_PATH "")
if (NOT ${XACC_ROOT} MATCHES ${CMAKE_INSTALL_PREFIX}) 
  message(STATUS "WE ARE BUILDING WITH HOMEBREW")
  set (QCOR_APPEND_PLUGIN_PATH "${CMAKE_INSTALL_PREFIX}/plugins")
endif()

configure_file(qcor.in.py
            ${CMAKE_BINARY_DIR}/qcor.py)

add_library(${LIBRARY_NAME} SHARED ${SRC})
target_include_directories(${LIBRARY_NAME} PUBLIC . ${CMAKE_SOURCE_DIR}/runtime/jit
                                          ${CMAKE_SOURCE_DIR}/lib/qsim
                                          ${Python_INCLUDE_DIRS}
                                          ${XACC_ROOT}/include/pybind11/include)
set_target_properties(${LIBRARY_NAME} PROPERTIES PREFIX "")
target_link_libraries(${LIBRARY_NAME} PUBLIC qcor qcor-qsim qcor-jit xacc::xacc)

if(APPLE)
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "@loader_path/lib")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

install(FILES ${CMAKE_BINARY_DIR}/qcor.py DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS _pyqcor DESTINATION ${CMAKE_INSTALL_PREFIX})

if (QCOR_BUILD_TESTS)
  add_subdirectory(tests)
endif()