# QCOR

QCOR is a C++ language extension and associated compiler implementation
for variational quantum computation on near-term, noisy devices.


## Dependencies
Compiler (C++11): GNU 5+, Clang 3+
CMake 3.9+
XACC: see https://xacc.readthedocs.io/en/latest/install.html#building-xacc

## Build instructions
For CMake 3.9+, do not use the apt-get installer, instead use `pip`, and
ensure that `/usr/local/bin` is in your PATH:
```bash
$ python -m pip install --upgrade cmake
$ export PATH=$PATH:/usr/local/bin
```

On Ubuntu 16+, install latest clang and llvm libraries and headers (you may need sudo)
```bash
$ wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
$ add-apt-repository "deb http://apt.llvm.org/jessie/ llvm-toolchain-jessie main"
$ apt-get update
$ apt-get install libclang-9-dev llvm-9-dev clang-9
$ ln -s /usr/bin/llvm-config-9 /usr/bin/llvm-config
```

Note that, for now, developers must clone QCOR manually:
``` bash
$ git clone https://code.ornl.gov/qci/qcor
$ cd qcor
$ mkdir build && cd build
$ cmake .. -DXACC_DIR=~/.xacc (or wherever you installed XACC)
$ make install
```
Update your PATH to ensure that the ```qcor``` compiler is available.
```bash
$ export PATH=$PATH:$HOME/.xacc/bin
```

To target IBM, Rigetti, or TNQVM, please also build the
corresponding XACC plugins. See https://xacc.readthedocs.io/en/latest/plugins.html.

## Example Usage

Here we demonstrate how to program, compile, and run the Deuteron H2 VQE problem. Create
the following file

```cpp
#include "qcor.hpp"

int main() {

  // Initialize the QCOR Runtime
  qcor::Initialize({"--accelerator", "tnqvm"});

  // Create an Optimizer, default is NLOpt COBYLA
  auto optimizer = qcor::getOptimizer("nlopt");

  // Create the Deuteron Observable
  const std::string deuteronH2 =
      R"deuteronH2((5.907,0) + (-2.1433,0) X0 X1 + (-2.1433,0) Y0 Y1 + (.21829,0) Z0 + (-6.125,0) Z1)deuteronH2";
  PauliOperator op;
  op.fromString(deuteronH2);

  // Schedule an asynchronous VQE execution
  // with the given quantum kernel ansatz
  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(
        [&](double t0) {
          X(0);
          Ry(t0, 1);
          CX(1, 0);
        },
        op, optimizer);
  });

  // Get and print the results
  auto results = future.get();
  results->print();
}
```
To compile this with QCOR, run the following

```bash
$ qcor deuteron.cpp -o deuteron
```
This will create the ```deuteron``` quantum-classical binary executable.
Now just run
```bash
$ ./deuteron
```