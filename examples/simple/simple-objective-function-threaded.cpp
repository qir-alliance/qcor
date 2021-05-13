// Note no includes here, we are just
// using the language extension
//
// run this with
// qcor -qpu qpp simple-objective-function-threaded.cpp
// ./a.out

// for _XACC_MUTEX
#include "xacc.hpp"

#ifdef SERIALIZE_PRINT
std::mutex print_lock;
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
  auto objective = createObjectiveFunction(
      ansatz, H, q, n_variational_params,
      {{"gradient-strategy", "parameter-shift"}});

  // Create the Optimizer.
  auto optimizer = createOptimizer("nlopt", {{"nlopt-optimizer", "l-bfgs"}});

  // Launch the Optimization Task with taskInitiate
  auto handle = taskInitiate(objective, optimizer);

  // Go do other work...

  // Query results when ready.
  auto results = sync(handle);
  printf("vqe-energy from taskInitiate = %f\n", results.opt_val);
#ifdef SERIALIZE_PRINT
  std::lock_guard<std::mutex> lock(print_lock);
#endif
  for (auto &x : linspace(-constants::pi, constants::pi, 20)) {
    std::cout << x << ", " << (*objective)({x}) << "\n";
  }
}

int main(int argc, char** argv) {
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
