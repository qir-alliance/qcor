# QCOR

QCOR is a C++ language extension and associated compiler implementation
for variational quantum computation on near-term, noisy devices.


## Dependencies
Compiler (C++11): GNU 5+, Clang 8+
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
$ echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main" > /etc/apt/sources.list.d/llvm.list
$ apt-get update
$ apt-get install -y libclang-9-dev llvm-9-dev
$ ln -s /usr/bin/llvm-config-9 /usr/bin/llvm-config
```

Note that, for now, developers must clone QCOR manually:
``` bash
$ git clone https://github.com/ornl-qci/qcor
$ cd qcor
$ mkdir build && cd build
$ cmake .. -DXACC_DIR=~/.xacc (or wherever you installed XACC)
$ make install
```
Update your PATH to ensure that the ```qcor``` compiler is available.
```bash
$ export PATH=$PATH:$HOME/.xacc/bin
```

## Example Usage

Here we demonstrate how to program, compile, and run the Deuteron H2 VQE problem. Create
the following file

```cpp
#include "qcor.hpp"

int main(int argc, char** argv) {

  // Initialize the QCOR Runtime
  qcor::Initialize(argc, argv);
  
  auto optimizer = qcor::getOptimizer(
      "nlopt", {{"nlopt-optimizer", "cobyla"},
                {"nlopt-maxeval", 20}});

  auto op = qcor::getObservable("pauli", "5.907 - 2.1433 X0X1 "
                                         "- 2.1433 Y0Y1"
                                         "+ .21829 Z0 - 6.125 Z1");

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
