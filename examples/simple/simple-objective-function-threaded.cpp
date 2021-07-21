// Note no includes here, we are just
// using the language extension
//
// run this with
// qcor -qpu qpp simple-objective-function-threaded.cpp
// ./a.out

// for _QCOR_MUTEX
#include "qcor_config.hpp"
#ifdef _QCOR_MUTEX
#include <mutex>
#include <thread>
#endif

__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}

void foo() {
  // Allocate 2 qubits
  auto q = qalloc(2);

  // Programmer needs to set 
  // the number of variational params
  auto n_variational_params = 1;

  // Create the Deuteron Hamiltonian
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);

  // Create the ObjectiveFunction, here we want to run VQE
  // need to provide ansatz, Operator, and qreg
  auto objective = createObjectiveFunction(ansatz, H, q, n_variational_params,
                                           {{"gradient-strategy", "central"}, {"step", 1e-3}});

  // Create the Optimizer.
  auto optimizer = createOptimizer("nlopt", {{"nlopt-optimizer", "l-bfgs"}});

  // Launch the Optimization Task with taskInitiate
  // auto handle = taskInitiate(objective, optimizer);
  auto [opt_val, opt_params] = optimizer->optimize(objective);

  // Go do other work...

  // Query results when ready.
  // auto results = sync(handle);
  // printf("vqe-energy from taskInitiate = %f\n", results.opt_val);
  qcor_expect(std::abs(opt_val + 1.74886) < 0.1);
  
}

int main(int argc, char** argv) {
#ifdef _QCOR_MUTEX
    std::cout << "_QCOR_MUTEX is defined: multi-threding execution" << std::endl;
    std::thread t0(foo);
    std::thread t1(foo);
    t0.join();
    t1.join();
#else
    std::cout << "_QCOR_MUTEX is NOT defined: sequential execution" << std::endl;
    foo();
    foo();
#endif
}
