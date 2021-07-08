set(EXTRA_INCLUDE "")
foreach(EXTRA_HEADER ${QCOR_EXTRA_HEADERS})
  set(EXTRA_INCLUDE "${EXTRA_INCLUDE} -I ${EXTRA_HEADER}")
endforeach()

configure_file("${CMAKE_SOURCE_DIR}/scripts/debian/postinst"
               "${CMAKE_BINARY_DIR}/scripts/debian/postinst")