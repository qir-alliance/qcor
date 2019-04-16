# A light-weight wrapper for LibInt

## Dependencies
```
Compiler (C++11, Fortran-2003): GNU 5+, Clang 3+
LibInt: https://github.com/evaleev/libint
CMake 3.9+ (for build)

## Build instructions
```
First, build LibInt
```bash
$ apt-get install libboost-dev
$ git clone https://github.com/evaleev/libint
$ cd libint && mkdir build && sh autogen.sh && cd build
$ CXXFLAGS=-fPIC CFLAGS=-fPIC ../configure
$ make install
```
Now, build libint-wrapper
```bash
$ mkdir build && cd build
$ cmake .. -DLIBINT_ROOT_DIR=/path/to/libint/install -DLIBINT_BUILD_TESTS=TRUE
$ make install
```

## Testing instructions
From build directory:
```bash
$ ctest (or ./src/tests/LibIntWrapperTester to run the executable)
```
