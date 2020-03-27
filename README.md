| Branch | Status |
|:-------|:-------|
|master | [![pipeline status](https://code.ornl.gov/qci/qcor/badges/master/pipeline.svg)](https://code.ornl.gov/qci/qcor/commits/master) |
|devel | [![pipeline status](https://code.ornl.gov/qci/qcor/badges/devel/pipeline.svg)](https://code.ornl.gov/qci/qcor/commits/devel) |

# QCOR

QCOR is a C++ language extension and associated compiler implementation
for hybrid quantum-classical programming.


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
$ cmake -G Ninja ../llvm -DCMAKE_INSTALL_PREFIX=$HOME/.llvm -DBUILD_SHARED_LIBS=TRUE -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_ENABLE_PROJECTS=clang
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

[[clang::syntax(xasm)]] void ansatz(qreg q, double t) {
  X(q[0]);
  Ry(q[1], t);
  CX(q[1], q[0]);
}

int main(int argc, char **argv) {

  auto opt = qcor::getOptimizer();
  auto obs = qcor::getObservable(
      "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1");

  // Schedule an asynchronous VQE execution
  // with the given quantum kernel ansatz
  auto handle = qcor::taskInitiateWithSyntax(ansatz, "vqe", opt, obs, 0.45);

  auto results_buffer = handle.get();
  auto energy = qcor::extract_results<double>(results_buffer, "opt-val");
  auto angles =
      qcor::extract_results<std::vector<double>>(results_buffer, "opt-params");

  printf("energy = %f\n", energy);
  printf("angles = [");
  for (int i = 0; i < 1; i++)
    printf("%f ", angles[i]);
  printf("]\n");
}
```
To compile this with QCOR targeting a Rigetti QCS QPU, run the following

```bash
$ qcor -o deuteron -a qcs:Aspen-4-4Q-A deuteron.cpp
```
This will create the ```deuteron``` quantum-classical binary executable.
Now just run
```bash
$ ./deuteron
```
