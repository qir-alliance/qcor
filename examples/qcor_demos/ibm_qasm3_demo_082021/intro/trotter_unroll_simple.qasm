// Show off quantum optimizations induced by Loop Unrolling
//
// Show unoptimized (note the affine.for)
// qcor --emit-mlir trotter_unroll_simple.qasm
// 
// Show optimized mlir (note affine.for removed)
// qcor --emit-mlir --q-optimize simple_opts.qasm
//
// Show unoptimized LLVM/QIR
// qcor --emit-llvm -O0 simple_opts.qasm
//
// Show optimized LLVM/QIR
// qcor --emit-llvm -O3 simple_opts.qasm

OPENQASM 3;
qubit qq[2];
for i in [0:100] {
    h qq;
    cx qq[0], qq[1];
    rx(0.0123) qq[1];
    cx qq[0], qq[1];
    h qq;
}