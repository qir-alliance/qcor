| Branch | Status |
|:-------|:-------|
|master | [![pipeline status](https://code.ornl.gov/qci/qcor/badges/master/pipeline.svg)](https://code.ornl.gov/qci/qcor/commits/master) |
|devel | [![pipeline status](https://code.ornl.gov/qci/qcor/badges/devel/pipeline.svg)](https://code.ornl.gov/qci/qcor/commits/devel) |

# QCOR

QCOR is a C++ language extension and associated compiler implementation
for variational quantum computation on near-term, noisy devices.


## Dependencies
```
Compiler (C++14): GNU 6.1+, Clang 3.4+
CMake 3.9+ (for build)
XACC: see https://xacc.readthedocs.io/en/latest/install.html#building-xacc
clang-dev version 8+
```
## Linux Build Instructions
Easiest way to install CMake - do not use the package manager,
instead use `pip`, and ensure that `/usr/local/bin` is in your PATH:
```bash
$ python -m pip install --upgrade cmake
$ export PATH=$PATH:/usr/local/bin
```

On Ubuntu 16.04 (for others, replace xenial with Ubuntu release name),
install latest clang and llvm libraries and headers (you may need sudo)
```bash
$ wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
$ echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-9 main" > /etc/apt/sources.list.d/llvm.list
$ apt-get update && apt-get install -y libclang-9-dev llvm-9-dev clangd-9
$ ln -s /usr/bin/llvm-config-9 /usr/bin/llvm-config
$ (for theia ide) ln -s /usr/bin/clangd-9 /usr/bin/clangd
```

Note that, for now, developers must clone QCOR manually:
``` bash
$ git clone https://github.com/ornl-qci/qcor
$ cd qcor
$ mkdir build && cd build
$ cmake .. -DXACC_DIR=~/.xacc (or wherever you installed XACC)
$ [with tests] cmake .. -DXACC_DIR=~/.xacc -DQCOR_BUILD_TESTS=TRUE
$ make install
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

int main(int argc, char **argv) {

  // Initialize QCOR
  qcor::Initialize(argc, argv);

  // Define your quantum kernel, here as a
  // standard C++ lambda containing quantum code
  auto ansatz = [&](qbit q, std::vector<double> t) {
    X(q[0]);
    Ry(q[1], t[0]);
    CNOT(q[1], q[0]);
  };

  // Get a valid Optimizer
  auto optimizer =
      qcor::getOptimizer("nlopt", {std::make_pair("nlopt-optimizer", "cobyla"),
                                   std::make_pair("nlopt-maxeval", 20)});

  // Define the Observable
  auto observable =
      qcor::getObservable("pauli", std::string("5.907 - 2.1433 X0X1 "
                                               "- 2.1433 Y0Y1"
                                               "+ .21829 Z0 - 6.125 Z1"));

  // Call qcor::taskInitiate to kick off asynchronous execution of
  // VQE with given ansatz, optimizer, and observable, and initial params 0.0
  auto handle = qcor::taskInitiate(ansatz, "vqe", optimizer, observable,
                                   std::vector<double>{0.0});

  // Go do other work, task is running asynchronously

  // Now request the results, this will wait
  // until the task finishes.
  auto results = qcor::sync(handle);

  // Get the results...
  std::cout << results->getInformation("opt-val").as<double>() << "\n";

  // Finalize the framework.
  qcor::Finalize();
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
