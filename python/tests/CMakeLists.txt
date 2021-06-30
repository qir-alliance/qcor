#add_test (NAME qcor_quasimo_python_bindings
#  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_quasimo.py
#  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#)
#set_tests_properties(qcor_quasimo_python_bindings
#    PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")

add_test (NAME qcor_python_jit_simple
 COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_jit_simple.py 
 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(qcor_python_jit_simple
   PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")

add_test (NAME qcor_python_jit_nested_kernels
COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_jit_nested.py 
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(qcor_python_jit_nested_kernels
  PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")

add_test (NAME qcor_python_kernel_builder
COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_kernel_builder.py 
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(qcor_python_kernel_builder
  PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")

add_test (NAME qcor_python_jit_decompose
 COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_jit_decompose.py 
 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(qcor_python_jit_decompose
   PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")

add_test (NAME qcor_test_qcor_spec_api
   COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_qcor_spec_api.py 
   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
 )
 set_tests_properties(qcor_test_qcor_spec_api
     PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")

add_test (NAME qcor_kernel_ftqc_python
 COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_kernel_ftqc.py -qrt ftqc
 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(qcor_kernel_ftqc_python
   PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")

add_test (NAME qcor_python_jit_pass_manager
   COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_jit_pass_manager.py
   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
 )
 set_tests_properties(qcor_python_jit_pass_manager
     PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")

add_test (NAME qcor_python_jit_multi_ctrl
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_jit_multi_ctrl.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(qcor_python_jit_multi_ctrl
  PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")

add_test (NAME qcor_python_jit_grover
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_jit_grover.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(qcor_python_jit_grover
  PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")

add_test (NAME qcor_python_jit_kernel_signature
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_jit_kernel_signature.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(qcor_python_jit_kernel_signature
  PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")

add_test (NAME qcor_python_compute_action
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_compute_action.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(qcor_python_compute_action
  PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")
 
add_test (NAME qcor_python_bug_89
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_jit_find_kernel_function_bug_89.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(qcor_python_bug_89
  PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")
 
add_test (NAME qcor_python_kernel_autograd
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_autograd_opt.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(qcor_python_kernel_autograd
  PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}")