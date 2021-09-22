# TNQVM Random Circuit Run

- Platform: Andes Cluster: 2 AMD EPYC 7302 16Core Processor 3.0 GHz, 16 cores (total 32 cores per node)

- Modules

```
module load gcc/10.3.0 cmake/3.18.4 python/3.7-anaconda3 git/2.29.0 openblas/0.3.12-omp
```

- ExaTN build (MPI enabled)
```
CC=gcc CXX=g++ FC=gfortran cmake .. -DCMAKE_BUILD_TYPE=Release -DEXATN_BUILD_TESTS=TRUE -DBLAS_LIB=OPENBLAS -DBLAS_PATH=/sw/andes/spack-envs/base/opt/linux-rhel8-x86_64/gcc-10.3.0/openblas-0.3.12-lvqlwh4l3ywjy3fmrfcusmh2ooyt2r4b/lib -DMPI_LIB=OPENMPI -DMPI_ROOT_DIR=/sw/andes/spack-envs/base/opt/linux-rhel8-x86_64/gcc-10.3.0/openmpi-4.0.4-4gclv46mxmuq3kriamaxinzezppb7vyi -DCMAKE_INSTALL_PREFIX=/ccs/proj/phy149/Thien/.exatn
```

- XACC and TNQVM build:

```
cmake .. -DCMAKE_INSTALL_PREFIX=/ccs/proj/phy149/Thien/.xacc
```

```
CC=gcc CXX=g++ FC=gfortran cmake .. -DCMAKE_BUILD_TYPE=Release -DXACC_DIR=/ccs/proj/phy149/Thien/.xacc -DEXATN_DIR=/ccs/proj/phy149/Thien/.exatn  -DTNQVM_BUILD_TESTS=TRUE -DTNQVM_BUILD_EXAMPLES=TRUE
```

- QCOR build
```
CC=gcc CXX=g++ cmake .. -DXACC_DIR=/ccs/proj/phy149/Thien/.xacc -DLLVM_ROOT=/ccs/proj/phy149/Thien/.llvm -DMLIR_DIR=/ccs/proj/phy149/Thien/.llvm/lib/cmake/mlir -DQCOR_BUILD_TESTS=TRUE -DQCOR_EXTRA_HEADERS="/sw/andes/gcc/10.3.0/include/c++/10.3.0/x86_64-pc-linux-gnu/;/sw/andes/gcc/10.3.0/include/c++/10.3.0/"
```

- Export PATH (for qcor command-line tools)

```
export PATH=/ccs/proj/phy149/Thien/.xacc/bin:$PATH
```

- Interactive session request (2 full node for 1 hour):

```
salloc -A PHYXXX -N 2 -t 1:00:00
```

- Run MPI: 4 tasks on 2 nodes, each uses 16 cores.

```
srun -n4 -N2 -c16 --cpu-bind=threads ./a.out -qrt nisq -qpu tnqvm -qpu-config tnqvm.ini
```

# DM-SIM Adder Circuit (GPU)

- Platform: Summit. 
Login node: (2) 16-core Power9 CPUs and (4) V100 GPUs. 
Compute nodes have (2) 22-core Power9 CPUs and (6) V100 GPUs.

Note: on login node, we can only use **one** GPU for testing.

- Request an interactive session: 

```
bsub -Is -W 1:00 -nnodes 1 -P PHYXXX $SHELL
```

Note: small jobs (e.g., single node) is low priority (w.r.t. Summit job scheduling), hence we may not be able to request
an allocation for the live session.