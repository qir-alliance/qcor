![qcor](docs/assets/qcor_full_logo.svg)

| master | 
|:-------|
| [![pipeline status](https://code.ornl.gov/qci/qcor/badges/master/pipeline.svg)](https://code.ornl.gov/qci/qcor/commits/master) |

# QCOR

QCOR is a C++ language extension and associated compiler implementation
for hybrid quantum-classical programming.


Documentation
-------------

* [Website and Documentation](https://qcor.readthedocs.io)
* [API Documentation](https://ornl-qci.github.io/qcor-api-docs/)

Quick Start
-----------
QCOR nightly docker images are available that serve up an Eclipse Theia IDE (the same IDE Gitpod uses) on port 3000. To get started, run 
```bash
$ docker run --security-opt seccomp=unconfined --init -it -p 3000:3000 qcor/qcor
```
Navigate to ``https://localhost:3000`` in your browser to open the IDE and get started with QCOR. 

## Dependencies
```
Compiler (C++14): GNU 6.1+, Clang 3.4+
CMake 3.9+ (for build)
XACC: see https://xacc.readthedocs.io/en/latest/install.html#building-xacc
LLVM/Clang [Syntax Handler Fork](https://github.com/hfinkel/llvm-project-csp).
```

## Linux Build Instructions
Easiest way to install CMake - do not use the package manager,
instead use `pip`, and ensure that `/usr/local/bin` is in your PATH:
```bash
$ python3 -m pip install --upgrade cmake
$ export PATH=$PATH:/usr/local/bin
```

For now we require our users build a specific fork of LLVM/Clang that 
provides Syntax Handler plugin support. We expect this fork to be upstreamed 
in a future release of LLVM and Clang, and at that point users will only 
need to download the appropriate LLVM/Clang binaries (via `apt-get` for instance).

To build this fork of LLVM/Clang (be aware this step takes up a good amount of RAM):
```bash
$ apt-get install ninja-build [if you dont have ninja]
$ git clone https://github.com/hfinkel/llvm-project-csp llvm
$ cd llvm && mkdir build && cd build
$ cmake -G Ninja ../llvm -DCMAKE_INSTALL_PREFIX=$HOME/.llvm -DBUILD_SHARED_LIBS=TRUE -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_ENABLE_DUMP=ON -DLLVM_ENABLE_PROJECTS=clang
$ cmake --build . --target install
$ sudo ln -s $HOME/.llvm/bin/llvm-config /usr/bin
```

Note that, for now, developers must clone QCOR manually:
``` bash
$ git clone https://github.com/ornl-qci/qcor
$ cd qcor
$ mkdir build && cd build
$ cmake .. 
$ [with tests] cmake .. -DQCOR_BUILD_TESTS=TRUE
$ make -j$(nproc) install
```
Update your PATH to ensure that the ```qcor``` compiler is available.
```bash
$ export PATH=$PATH:$HOME/.xacc/bin (or wherever you installed XACC)
```

## Example Usage

Here we demonstrate how to program, compile, and run the Deuteron H2 VQE problem. Create
the following file

```cpp
#include "qcor.hpp"

// QCOR kernel requirements:
// C-like function, unique function name
// takes qreg as first argument, can take any 
// arguments after that which are necessary for function 
// evaluation (like gate rotation parameters). 
// Function body is written in a 'supported language' (i.e. 
// we have a xacc::Compiler parser for it, here XASM (which is default))
// Must be annotated with the __qpu__ attribute, which expands 
// to [[clang::syntax(qcor)]], thereby invoking our custom Clang SyntaxHandler.

__qpu__ void ansatz(qreg q, double t) {
  X(q[0]);
  Ry(q[1], t);
  CX(q[1], q[0]);
}

int main(int argc, char **argv) {

 // Allocate 2 qubits
  auto q = qalloc(2);

  // Create the Deuteron Hamiltonian (Observable)
  auto H = qcor::createObservable(
      "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1");

  // Create the ObjectiveFunction, here we want to run VQE
  // need to provide ansatz and the Observable
  auto objective = qcor::createObjectiveFunction("vqe", ansatz, H);

  // Evaluate the ObjectiveFunction at a specified set of parameters
  auto energy = (*objective)(q, .59);

  // Print the result
  printf("vqe energy = %f\n", energy);
  q.print();

}
```
To compile this with QCOR targeting a Rigetti QCS QPU, run the following

```bash
$ [run on QPU] qcor -o deuteron_qcs -qpu qcs:Aspen-4-4Q-A deuteron.cpp
$ [run on Simulator] qcor -o deuteron_tnqvm -qpu tnqvm deuteron.cpp
```
This will create the ```deuteron_tnqvm``` and ```deuteron_qcs``` quantum-classical binary executables, 
each compiled for the specified backend. Now just run one of them
```bash
$ ./deuteron_tnqvm
```
