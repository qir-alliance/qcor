/*
 * quantum ripple-carry adder
 * Cuccaro et al, quant-ph/0410184
 */
// Summit compile:
// Summit set-up:
// module load cmake/3.20.2 python/3.8.10 gcc/9.3.0 cuda/11.4.0
// Interactive node request:
// bsub -Is -W 1:00 -nnodes 1 -P PHYXXX $SHELL
// Note: Summit can be **fully-booked** (running leadership-type jobs) hence the above
// request can be pending in a long time. 
// Using DM-Sim:
// qcor -linker g++ -qrt nisq adder.qasm -shots 1024 -qpu dm-sim[gpus:4]
// Note: on a login node,GPU-GPU comm is disabled, hence can only run with 1 GPU.


// Utils:
// See number of GPU's: nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
// XACC/LLVM-CSP/QCOR compile:
// LLVM-CSP
// cmake ../llvm -DCMAKE_INSTALL_PREFIX=~/.llvm -DCMAKE_C_COMPILER=/sw/summit/gcc/9.3.0-2/bin/gcc -DCMAKE_CXX_COMPILER=/sw/summit/gcc/9.3.0-2/bin/g++ -DLLVM_ENABLE_PROJECTS="clang;mlir" -DBUILD_SHARED_LIBS=TRUE

// QCOR: (must add paths to the specific gcc module paths)
// cmake .. -DXACC_DIR=~/.xacc -DLLVM_ROOT=~/.llvm -DMLIR_DIR=~/.llvm/lib/cmake/mlir -DQCOR_BUILD_TESTS=TRUE -DCMAKE_BUILD_TYPE=Debug -DQCOR_EXTRA_HEADERS="/sw/summit/gcc/9.3.0-2/include/c++/9.3.0/powerpc64le-unknown-linux-gnu/;/sw/summit/gcc/9.3.0-2/include/c++/9.3.0/"

// export PATH=~/.xacc/bin:$PATH

OPENQASM 3;

gate ccx a,b,c
{
  h c;
  cx b,c; tdg c;
  cx a,c; t c;
  cx b,c; tdg c;
  cx a,c; t b; t c; h c;
  cx a,b; t a; tdg b;
  cx a,b;
}

gate majority a, b, c {
  cx c, b;
  cx c, a;
  ccx a, b, c;
}

gate unmaj a, b, c {
  ccx a, b, c;
  cx c, a;
  cx a, b;
}

qubit cin;
qubit a[8];
qubit b[8];
qubit cout;
bit ans[9];

// FIXME: left shift casting to handle different type:
// e.g. i8 vs. i64 (shift value)
uint[64] a_in = 1;  
uint[64] b_in = 15; 

for i in [0:8] {
  // FIXME: not able to do this inline....
  bool b1 = bool(a_in[i]);
  bool b2 = bool(b_in[i]);
  if (b1) {
    x a[i];
  }
  if (b2) {
    x b[i];
  }
}
// add a to b, storing result in b
majority cin, b[0], a[0];

for i in [0: 7] { 
  majority a[i], b[i + 1], a[i + 1]; 
}

cx a[7], cout;

for i in [6: -1: -1] { 
  unmaj a[i], b[i+1], a[i+1]; 
}
unmaj cin, b[0], a[0];

measure b[0:7] -> ans[0:7];
measure cout[0] -> ans[8];