// Note no includes here, we are just
// using the language extension
//
// run this with
// qcor -qpu qpp simple-objective-function-async.cpp
// ./a.out

// for _QCOR_MUTEX
#include "qcor_config.hpp"
#ifdef _QCOR_MUTEX
#include <mutex>
#include <future>
#include <thread>
#endif

__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}

int main(int argc, char **argv) {
#ifdef _QCOR_MUTEX
    std::cout << "_QCOR_MUTEX is defined: execute taskInitiate asynchronously" << std::endl;
#else
    std::cout << "_QCOR_MUTEX is NOT defined: execute taskInitiate sequentially" << std::endl;
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
  auto objective1 = createObjectiveFunction(ansatz, H1, q1, n_variational_params1,
                                            {{"gradient-strategy", "central"}, {"step", 1e-3}});
  auto objective2 = createObjectiveFunction(ansatz, H2, q2, n_variational_params2,
                                            {{"gradient-strategy", "central"}, {"step", 1e-3}});

  // Create the Optimizer.
  auto optimizer1 = createOptimizer("nlopt", {{"nlopt-optimizer", "l-bfgs"}});
  auto optimizer2 = createOptimizer("nlopt", {{"nlopt-optimizer", "l-bfgs"}});

#ifdef _QCOR_MUTEX
  // Launch the two optimizations asynchronously
  auto handle1 = std::async(std::launch::async, [=]() -> std::pair<double, std::vector<double>> { return optimizer1->optimize(objective1); });
  auto handle2 = std::async(std::launch::async, [=]() -> std::pair<double, std::vector<double>> { return optimizer2->optimize(objective2); });

  // Go do other work...

  // Query results when ready.
  auto [opt_val1, opt_params1] = handle1.get();
  auto [opt_val2, opt_params2] = handle2.get();
#else
  // Launch the two optimizations sequentially
  auto [opt_val1, opt_params1] = optimizer1->optimize(objective1);
  auto [opt_val2, opt_params2] = optimizer2->optimize(objective2);
#endif
  qcor_expect(std::abs(opt_val1 + 1.74886) < 0.1);
  qcor_expect(std::abs(opt_val2 + 1.74886) < 0.1);
  
}