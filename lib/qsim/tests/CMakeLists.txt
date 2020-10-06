file(GLOB SRC *.cpp)

add_executable(test-qsim ${SRC})
target_include_directories(test-qsim PUBLIC ..)
target_link_libraries(test-qsim PUBLIC qcor qcor-qsim)
