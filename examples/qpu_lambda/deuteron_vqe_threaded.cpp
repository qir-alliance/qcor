#include "qcor.hpp"
// for _XACC_MUTEX
#include "xacc.hpp"

void foo() {
  // Create the Hamiltonian
  auto H = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1) + 5.907;

  auto ansatz = qpu_lambda([](qreg q, double x) {
    print("x = ", x);
    X(q[0]);
    Ry(q[1], x);
    CX(q[1], q[0]);
  });

  auto ansatz_take_vec = qpu_lambda([](qreg q, std::vector<double> x) {
    print("x = ", x[0]);
    X(q[0]);
    Ry(q[1], x[0]);
    CX(q[1], q[0]);
  });

  OptFunction opt_function(
      [&](std::vector<double> x) { return ansatz.observe(H, qalloc(2), x[0]); },
      1);

  OptFunction opt_function_vec(
      [&](std::vector<double> x) {
        return ansatz_take_vec.observe(H, qalloc(2), x);
      },
      1);

  auto optimizer = createOptimizer("nlopt");
  auto [ground_energy, opt_params] = optimizer->optimize(opt_function);
  print("Energy: ", ground_energy);
  qcor_expect(std::abs(ground_energy + 1.74886) < 0.1);

  auto [ground_energy_vec, opt_params_vec] =
      optimizer->optimize(opt_function_vec);
  print("Energy: ", ground_energy_vec);
  qcor_expect(std::abs(ground_energy_vec + 1.74886) < 0.1);
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