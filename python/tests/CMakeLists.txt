add_test (NAME qcor_qsim_python_bindings
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_qsim.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)