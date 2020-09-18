#include "qcor_hybrid.hpp"

// Define one quantum kernel that takes
// double angle parameter
__qpu__ void ansatz(qreg q, double x) {
  X(q[0]);
  Ry(q[1], x);
  CX(q[1], q[0]);
}

int main() {

  // Define the Hamiltonian using the QCOR API
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);

  // Create a VQE instance, must give it
  // the parameterized ansatz functor and Observable
  VQE vqe(ansatz, H);
  for (auto [iter, x] :
       enumerate(linspace(-constants::pi, constants::pi, 20))) {
    std::cout << iter << ", " << x << ", " << vqe({x}) << "\n";
  }

  vqe.persist_data("param_sweep_data.json");
}