// Note no includes here, we are just
// using the language extension
//
// run this with
// qcor -qpu qpp simple-objective-function-async.cpp
// ./a.out

// for _XACC_MUTEX
#include "xacc.hpp"

__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}

int main(int argc, char **argv) {
#ifdef _XACC_MUTEX
    std::cout << "_XACC_MUTEX is defined: execute taskInitiate asynchronously" << std::endl;
#else
    std::cout << "_XACC_MUTEX is NOT defined: execute taskInitiate sequentially" << std::endl;
#endif
  // Allocate 2 qubits
  auto q1 = qalloc(2);
  auto q2 = qalloc(2);

  // Programmer needs to set
  // the number of variational params
  auto n_variational_params1 = 1;
  auto n_variational_params2 = 1;

  // Create the Deuteron Hamiltonian
  auto H1 = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);
  auto H2 = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);

  // Create the ObjectiveFunction, here we want to run VQE
  // need to provide ansatz, Operator, and qreg
  auto objective1 = createObjectiveFunction(
      ansatz, H1, q1, n_variational_params1,
      {{"gradient-strategy", "parameter-shift"}});
  auto objective2 = createObjectiveFunction(
      ansatz, H2, q2, n_variational_params2,
      {{"gradient-strategy", "parameter-shift"}});

  // Create the Optimizer.
  auto optimizer1 = createOptimizer("nlopt", {{"nlopt-optimizer", "l-bfgs"}});
  auto optimizer2 = createOptimizer("nlopt", {{"nlopt-optimizer", "l-bfgs"}});

#ifdef _XACC_MUTEX
  // Launch the Optimization Task with taskInitiate
  auto handle1 = taskInitiate(objective1, optimizer1);
  // Go do other work...
  auto handle2 = taskInitiate(objective2, optimizer2);

  // Query results when ready.
  auto results1 = sync(handle1);
  auto results2 = sync(handle2);
#else
  // Launch the Optimization Task with taskInitiate
  auto handle1 = taskInitiate(objective1, optimizer1);
  // Query results when ready.
  auto results1 = sync(handle1);

  // Launch the Optimization Task with taskInitiate
  auto handle2 = taskInitiate(objective2, optimizer2);
  // Query results when ready.
  auto results2 = sync(handle2);
#endif
  printf("vqe-energy from taskInitiate1 = %f\n", results1.opt_val);
  printf("vqe-energy from taskInitiate2 = %f\n", results2.opt_val);

  printf("From objetive1\n");
  for (auto &x : linspace(-constants::pi, constants::pi, 20)) {
    std::cout << x << ", " << (*objective1)({x}) << "\n";
  }
  printf("From objetive2\n");
  for (auto &x : linspace(-constants::pi, constants::pi, 20)) {
    std::cout << x << ", " << (*objective2)({x}) << "\n";
  }
}