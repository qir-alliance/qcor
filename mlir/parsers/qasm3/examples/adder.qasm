/*
 * quantum ripple-carry adder
 * Cuccaro et al, quant-ph/0410184
 */
// Summit compile:
// Summit set-up:
// module load cmake/3.20.2 python/3.8.10 gcc/9.3.0 cuda/11.4.0 openblas/0.3.15-omp
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

// EXATN and TNQVM Build:
// CC=gcc CXX=g++ FC=gfortran cmake .. -DEXATN_BUILD_TESTS=TRUE -DBLAS_LIB=OPENBLAS -DBLAS_PATH=/sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-9.3.0/openblas-0.3.15-ydivokmxgbws566z3akrpxovkwm3rkcr/lib -DMPI_LIB=OPENMPI -DMPI_ROOT_DIR=/sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-9.3.0/spectrum-mpi-10.4.0.3-20210112-2s7kpbzydf6val7k2d3e6cz3zdhtcwlw -DENABLE_CUDA=True -DCUDA_HOST_COMPILER=/sw/summit/gcc/9.3.0-2/bin/g++ -DCMAKE_INSTALL_PREFIX=~/.exatn

// cmake .. -DXACC_DIR=~/.xacc -DEXATN_DIR=~/.exatn

// export PATH=~/.xacc/bin:$PATH

// Install DM-SIM plugin: qcor -install-plugin https://github.com/ORNL-QCI/DM-Sim.git

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
// Input values:
uint[8] a_in = 1;  
uint[8] b_in = 15; 

for i in [0:8] {  
  if (bool(a_in[i])) {
    x a[i];
  }
  if (bool(b_in[i])) {
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