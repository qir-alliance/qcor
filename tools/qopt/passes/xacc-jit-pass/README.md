## Example Usage

```cpp
#include "qcor.hpp"
#include <iostream>
#include <qalloc>

__qpu__ void f(qreg q) {
  H(q[0]);
  H(q[0]);
  H(q[0]);
  CX(q[0], q[1]);
  CX(q[0], q[1]);
  CX(q[0], q[1]);
  Measure(q[0]);
  Measure(q[1]);
}
int main() {
  auto q = qalloc(2);
  f(q);
  std::cout << "compiler optimized:\n";
  qcor::print_kernel(std::cout, f, q);
}
```

to compile and optimize manually, run the following 

```bash 
$ qcor -v -qrt -emit-llvm -c test.cpp 
$ qopt -xacc-optimize test.bc -o opted2.bc 
$ llc-9 -filetype=obj opted2.bc 
$ clang++ -Wl,-rpath,/home/cades/.xacc/lib -L /home/cades/.xacc/lib -lxacc -lqrt -lqcor -lxacc-quantum-gate -lxacc-pauli opted2.o -o out.x 
$ ./out.x 