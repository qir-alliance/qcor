#include "qcor.hpp"
// for _QCOR_MUTEX
#include "qcor_config.hpp"
#ifdef _QCOR_MUTEX
#include <mutex>
#include <thread>
#endif

void foo() {
  // Create the Hamiltonian
  auto H = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1) + 5.907;
  int iter_count = 0;
  auto ansatz = qpu_lambda(
      [](qreg q, double x) {
        X(q[0]);
        Ry(q[1], x);
        CX(q[1], q[0]);
        print("Iter", iter_count, "; angle = ", x);
        iter_count++;
      },
      iter_count);

  auto q = qalloc(2);
  auto objective = createObjectiveFunction(ansatz, H, q, 1);
  // Create a qcor Optimizer
  auto optimizer = createOptimizer("nlopt");

  // Optimize the above function
  auto [optval, opt_params] = optimizer->optimize(*objective.get());
  std::cout << "Energy: " << optval << "\n";
  qcor_expect(std::abs(optval + 1.74886) < 0.1);

  auto ansatz_vec_param = qpu_lambda([](qreg q, std::vector<double> x) {
    X(q[0]);
    Ry(q[1], x[0]);
    CX(q[1], q[0]);
  });

  auto q1 = qalloc(2);
  auto objective_vec = createObjectiveFunction(ansatz_vec_param, H, q1, 1);

  // Optimize the above function
  auto [optval_vec, opt_params_vec] = optimizer->optimize(*objective_vec.get());
  std::cout << "Energy: " << optval_vec << "\n";
  qcor_expect(std::abs(optval_vec + 1.74886) < 0.1);
}

int main(int argc, char **argv) {
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
