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

To target IBM, Rigetti, or TNQVM, please also build the
corresponding XACC plugins. See https://xacc.readthedocs.io/en/latest/plugins.html.