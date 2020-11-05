![qcor](docs/assets/qcor_full_logo.svg)

| master | 
|:-------|
| [![pipeline status](https://code.ornl.gov/qci/qcor/badges/master/pipeline.svg)](https://code.ornl.gov/qci/qcor/commits/master) |

# QCOR

QCOR is a C++ language extension and associated compiler implementation
for hybrid quantum-classical programming.


Documentation
-------------

* [Website and Documentation](https://aide-qc.github.io/deploy)
* [API Documentation](https://ornl-qci.github.io/qcor-api-docs/)

Quick Start
-----------
QCOR is available via pre-built deb packages on Ubuntu Bionic (18.04) and Focal (20.04). To install on Bionic
```bash
wget -qO- https://aide-qc.github.io/deploy/aide_qc/debian/PUBLIC-KEY.gpg | sudo apt-key add -
sudo wget -qO- "https://aide-qc.github.io/deploy/aide_qc/debian/$(lsb_release -cs)/aide-qc.list" > /etc/apt/sources.list.d/aide-qc.list
sudo apt-get update
sudo apt-get install qcor
```
Note that the above requires you have `lsb_release` installed (usually is, if not, `apt-get install lsb-release`).

QCOR nightly docker images are available that serve up an Eclipse Theia IDE (the same IDE Gitpod uses) on port 3000. To get started, run 
```bash
docker run --security-opt seccomp=unconfined --init -it -p 3000:3000 qcor/qcor
```
Navigate to ``https://localhost:3000`` in your browser to open the IDE and get started with QCOR. 

## Dependencies
- Compiler (C++17): GNU 8.4+, Clang 5+
- CMake 3.12+ (for build)
- XACC - [Build XACC](https://xacc.readthedocs.io/en/latest/install.html#building-xacc).
- LLVM/Clang - [Syntax Handler Fork](https://github.com/hfinkel/llvm-project-csp).

## Linux Build Instructions
Easiest way to install CMake - do not use the package manager,
instead use `pip`, and ensure that `/usr/local/bin` is in your PATH:
```bash
python3 -m pip install --upgrade cmake
export PATH=$PATH:/usr/local/bin
```

For now we require our users build a specific fork of LLVM/Clang that 
provides Syntax Handler plugin support. We expect this fork to be upstreamed 
in a future release of LLVM and Clang, and at that point users will only 
need to download the appropriate LLVM/Clang binaries (via `apt-get` for instance).

To build this fork of LLVM/Clang (be aware this step takes up a good amount of time / RAM):
```bash
apt-get install ninja-build [if you dont have ninja]
git clone https://github.com/hfinkel/llvm-project-csp llvm
cd llvm && mkdir build && cd build
cmake -G Ninja ../llvm -DCMAKE_INSTALL_PREFIX=$HOME/.llvm -DBUILD_SHARED_LIBS=TRUE -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_ENABLE_DUMP=ON -DLLVM_ENABLE_PROJECTS=clang
cmake --build . --target install
sudo ln -s $HOME/.llvm/bin/llvm-config /usr/bin
```

To build QCOR from source:
``` bash
git clone https://github.com/ornl-qci/qcor
cd qcor
mkdir build && cd build
[Running cmake with no flags will search for LLVM using `llvm-config` executable in your PATH and XACC in $HOME/.xacc]
cmake .. 
   [Extra optional flags] 
   -DQCOR_BUILD_TESTS=TRUE
   -DLLVM_ROOT=/path/to/llvm
   -DXACC_DIR=/path/to/xacc
make -j$(nproc) install
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
  auto H = createObservable(
      "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1");

  // Create the ObjectiveFunction, here we want to run VQE
  // need to provide ansatz and the Observable
  auto objective = createObjectiveFunction("vqe", ansatz, H);

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

## Cite QCOR 
If you use qcor in your research, please use the following citation 
```
@ARTICLE{qcor,
       author = {{Nguyen}, Thien and {Santana}, Anthony and {Kharazi}, Tyler and
         {Claudino}, Daniel and {Finkel}, Hal and {McCaskey}, Alexander},
        title = "{Extending C++ for Heterogeneous Quantum-Classical Computing}",
      journal = {arXiv e-prints},
     keywords = {Quantum Physics, Computer Science - Mathematical Software},
         year = 2020,
        month = oct,
          eid = {arXiv:2010.03935},
        pages = {arXiv:2010.03935},
archivePrefix = {arXiv},
       eprint = {2010.03935},
 primaryClass = {quant-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv201003935N},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```