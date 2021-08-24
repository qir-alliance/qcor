// Show off quantum optimizations induced by Loop Unrolling + Inlining
//
// Show unoptimized (note the affine.for)
// qcor --emit-mlir trotter_unroll_with_inlining.qasm
// 
// Show optimized mlir (note affine.for removed)
// qcor --emit-mlir trotter_unroll_with_inlining.qasm -O3
//
// Show unoptimized LLVM/QIR
// qcor --emit-llvm -O0 simple_opts.qasm
//
// Show optimized LLVM/QIR
// qcor --emit-llvm -O3 simple_opts.qasm

OPENQASM 3;
def cnot_ladder() qubit[4]:q {
  h q[0];
  h q[1];
  cx q[0], q[1];
  cx q[1], q[2];
  cx q[2], q[3];
}

def cnot_ladder_inv() qubit[4]:q {
  cx q[2], q[3];
  cx q[1], q[2];
  cx q[0], q[1];
  h q[1];
  h q[0];
}

qubit q[4];
double theta = 0.01;
for i in [0:100] {
  cnot_ladder q;
  rz(theta) q[3];
  cnot_ladder_inv q;
}