// Run this with 
// qcor all_x_gates.qasm 
// ./a.out 

// Can see mlir with 
// qcor-mlir-tool -emit=mlir all_x_gates.qasm

// Can see llvm ir with 
// qcor-mlir-tool -emit=llvm all_x_gates.qasm

OPENQASM 3;
include "qelib1.inc";

const n = 10;

qubit q[n];

for i in [0:n] {
  x q[i];
}

bit c[n];
c = measure q;

for i in [0:n] {
  print("bit result", i, "=", c[i]);
}