#include "qir-qrt.hpp"
using qubit = Qubit*;

[[clang::syntax(qasm3)]] int test(int i, qubit q, qubit r) {
  int ten = 10 * i;
  h q;
  cx q, r;
  bit c[2];
  c[0] = measure q;
  c[1] = measure r;
  reset q;
  reset r;
  return int[32](c);
}

// Run with...
// clang++ -std=c++17 -fplugin=/path/to/libqasm3-syntax-handler.so -I /path/to/qcor/include/qcor -I/path/to/xacc/include/xacc -c qasm3_test.cpp
// llc -filetype=obj test.bc (test comes from kernel name, so will need to do this for all kernels)
// clang++ -L /path/to/install/lib -lqir-qrt -lqcor -lxacc -lqrt -lCppMicroServices test.o qasm3_test.o
// ./a.out

int main(int argc, char** argv) {
  int x = 10;

  // Figure out how to initialize automatically
  __quantum__rt__initialize(argc, reinterpret_cast<int8_t**>(argv));
  
  
  // Connect this to qalloc(...)
  auto qreg = __quantum__rt__qubit_allocate_array(2);

  // Should be able to get qubit from operator[] on qreg
  auto qbit_mem = __quantum__rt__array_get_element_ptr_1d(qreg, 0);
  auto qbit = reinterpret_cast<Qubit**>(qbit_mem)[0];
  auto qbit_mem2 = __quantum__rt__array_get_element_ptr_1d(qreg, 1);
  auto qbit2 = reinterpret_cast<Qubit**>(qbit_mem2)[0];

  // Run bell test...
  int ones = 0, zeros = 0;
  for (int i = 0; i < 50; i++) {
    auto y = test(x, qbit, qbit2);
    // should be binary-as-int 00 = 0, or 11 = 3
    if (y == 3) {
      ones++;
    } else {
      zeros++;
    }
  }
  printf("Result: 11:%d, 00:%d\n", ones, zeros);
  return 0;
}