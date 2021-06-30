# QCOR HPC Simulator Backend

## Test cases: Sycamore random circuit sampling

## Goals

- Demonstrating system-level integration with HPC simulator backend (TNQVM/ExaTN on Summit) and remote Atos QLM.

- Configuring/targeting complex QPU backends via init files. 

## Outline

- Sycamore circuits: 53 qubits; print the circuit gate count, etc.

- Init file: configuring TNQVM (amplitude calculation mode). QCOR stack on Summit. Execution with MPI (showing GPU flops after calculation)

- Submit a job to QLM via a simple `-qpu` switch. Make sure `.qlm_config` is valid.
```
pip3 install qlmaas

qcor -qpu atos-qlm[sim-type:MPS] file.cpp
```


## Summit Build Instructions

### Modules

```
module load gcc/10.2.0 cmake/3.20.2 python/3.7.0 openblas/0.3.9-omp
```

### LLVM CSP Build

```
cmake ../llvm -DCMAKE_INSTALL_PREFIX=~/.llvm -DCMAKE_C_COMPILER=/sw/summit/gcc/10.2.0/bin/gcc -DCMAKE_CXX_COMPILER=/sw/summit/gcc/10.2.0/bin/g++ -DLLVM_ENABLE_PROJECTS="clang;mlir" -DBUILD_SHARED_LIBS=TRUE
```

### QCOR Build

```
cmake .. -DXACC_DIR=~/.xacc -DLLVM_ROOT=~/.llvm -DMLIR_DIR=~/.llvm/lib/cmake/mlir -DQCOR_BUILD_TESTS=TRUE -DCMAKE_BUILD_TYPE=Debug -DQCOR_EXTRA_HEADERS="/sw/summit/gcc/10.2.0/include/c++/10.2.0/powerpc64le-unknown-linux-gnu/;/sw/summit/gcc/10.2.0/include/c++/10.2.0/"
```

### Using QCOR

For some reason, Clang-CSP on Summit failed to link against libc++ loaded by module load (gcc/10.2).
Hence, we need to separate the QCOR compilation to two phases: (1) compile with qcor (-c) then (2) using g++ to link.

```
qcor -c file_name.cpp
```

```
g++ -rdynamic -Wl,-rpath,/ccs/home/nguyent/.xacc/lib:/ccs/home/nguyent/.xacc/lib:/autofs/nccs-svm1_home1/nguyent/.llvm/lib:/ccs/home/nguyent/.xacc/clang-plugins -L /ccs/home/nguyent/.xacc/lib -lqcor -lqrt -lqcor-quasimo -lqcor-jit -L /ccs/home/nguyent/.xacc/lib -lxacc -lCppMicroServices -lxacc-quantum-gate -lxacc-pauli -lxacc-fermion -lpthread -lqir-qrt file_name.o
```

### ExaTN Build

```
CC=gcc CXX=g++ FC=gfortran cmake .. -DCMAKE_BUILD_TYPE=Debug -DEXATN_BUILD_TESTS=TRUE -DBLAS_LIB=OPENBLAS -DBLAS_PATH=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-10.2.0/openblas-0.3.9-jzzdcglqreyi3t22emvurkxhrpkj4bhl/lib -DCMAKE_INSTALL_PREFIX=~/.exatn
```

### TNQVM Build

```
cmake .. -DXACC_DIR=~/.xacc -DEXATN_DIR=~/.exatn -DCMAKE_BUILD_TYPE=Debug 
```