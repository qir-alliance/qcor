set (PACKAGE_NAME "QCor Analog Instructions")
set (PACKAGE_DESCRIPTION "QCor Analog Instruction Bundle")
set (LIBRARY_NAME qcor-instructions)

file (GLOB_RECURSE HEADERS *.hpp)
file (GLOB SRC *.cpp ypulse/*.cpp ameas/*.cpp hwe/*.cpp)

# Set up dependencies to resources to track changes
usFunctionGetResourceSource(TARGET ${LIBRARY_NAME} OUT SRC)
# Generate bundle initialization code
usFunctionGenerateBundleInit(TARGET ${LIBRARY_NAME} OUT SRC)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ypulse)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ameas)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/hwe)

add_library(${LIBRARY_NAME} SHARED ${SRC})

set(_bundle_name qcor_instructions)

set_target_properties(${LIBRARY_NAME} PROPERTIES
    # This is required for every bundle
    COMPILE_DEFINITIONS US_BUNDLE_NAME=${_bundle_name}
    # This is for convenience, used by other CMake functions
    US_BUNDLE_NAME ${_bundle_name}
    )

# Embed meta-data from a manifest.json file
usFunctionEmbedResources(TARGET ${LIBRARY_NAME}
    WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
    manifest.json
    )

target_link_libraries(${LIBRARY_NAME} xacc xacc-quantum-gate xacc-pauli)

if(APPLE)
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "@loader_path/../lib")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)

if(QCOR_BUILD_TESTS)
   add_subdirectory(tests)
endif()