#!/bin/bash
set -x
@CMAKE_BINARY_DIR@/qcor -qrt -qpu aer -shots 1024 -emit-llvm -c @CMAKE_CURRENT_SOURCE_DIR@/test_opt.cpp
@LLVM_INSTALL_PREFIX@/bin/llc -filetype=obj @CMAKE_BINARY_DIR@/tools/qopt/passes/xacc-ir-transformation/tests/test_opt.bc
@LLVM_INSTALL_PREFIX@/bin/clang++ -Wl,-rpath,@XACC_ROOT@/lib -L @XACC_ROOT@/lib -lxacc -lqrt -lqcor -lxacc-quantum-gate -lxacc-pauli test_opt.o -o out.x 
./out.x