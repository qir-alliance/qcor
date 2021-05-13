// for _XACC_MUTEX
#include "xacc.hpp"

// Define the bell kernel
__qpu__ void bell(qreg q) {
  using qcor::xasm;
  H(q[0]);
  CX(q[0], q[1]);

  for (int i = 0; i < q.size(); i++) {
    Measure(q[i]);
  }
}

void foo() {
  // Create two qubit registers, each size 2
  auto q = qalloc(2);

  // Run the quantum kernel
  bell(q);

  // dump the results
  q.print();
}

int main(int argc, char **argv) {
#ifdef _XACC_MUTEX
    std::cout << "_XACC_MUTEX is defined: multi-threding execution" << std::endl;
    std::thread t0(foo);
    std::thread t1(foo);
    t0.join();
    t1.join();
#else
    std::cout << "_XACC_MUTEX is NOT defined: sequential execution" << std::endl;
    foo();
    foo();
#endif
}
