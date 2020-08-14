#include "qcor_hybrid.hpp"
#include "qcor_utils.hpp"
#include <iomanip>

// Define the state preparation kernel
__qpu__ void initial_state(qreg q) {
  X(q[0]);
  X(q[2]);
}

int main() {

  // Define the Hamiltonian using the QCOR API
  auto H = 0.1202 * Z(0) * Z(1) + 0.168336 * Z(0) * Z(2) +
           0.1202 * Z(2) * Z(3) + 0.17028 * Z(2) + 0.17028 * Z(0) +
           0.165607 * Z(0) * Z(3) + 0.0454063 * Y(0) * Y(1) * X(2) * X(3) -
           0.106477 - 0.220041 * Z(3) + 0.174073 * Z(1) * Z(3) +
           0.0454063 * Y(0) * Y(1) * Y(2) * Y(3) - 0.220041 * Z(1) +
           0.165607 * Z(1) * Z(2) + 0.0454063 * X(0) * X(1) * Y(2) * Y(3) +
           0.0454063 * X(0) * X(1) * X(2) * X(3);

  // optimizer
  auto optimizer = qcor::createOptimizer(
      "nlopt", {std::make_pair("nlopt-optimizer", "l-bfgs")});

  // Create ADAPT-VQE instance
  // Run H2 with the singlet-adapted-uccsd pool
  ADAPT adapt(initial_state, H, optimizer,
                    {{"sub-algorithm", "vqe"},
                     {"pool", "singlet-adapted-uccsd"},
                     {"n-electrons", 2},
                     {"gradient_strategy", "central"}});

  // Execute!
  auto energy = adapt.execute();

  // Print the energy
  std::cout << std::setprecision(12) << energy << "\n";
}
