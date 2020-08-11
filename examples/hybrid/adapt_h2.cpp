#include "qcor_hybrid.hpp"
#include <iomanip>

// Define the state preparation kernel
__qpu__ void initial_state(qreg q) {
  X(q[0]);
  X(q[2]);
}

int main() {

  // Define the Hamiltonian using the QCOR API
  std::string opstr =
      "(0.1202,0) Z0 Z1 + (0.168336,0) Z0 Z2 + (0.1202,0) Z2 Z3 + (0.17028,0) "
      "Z2 + (0.17028,0) Z0 + (0.165607,0) Z0 Z3 + (0.0454063,0) Y0 Y1 X2 X3 + "
      "(-0.106477,0) + (-0.220041,0) Z3 + (0.174073,0) Z1 Z3 + (0.0454063,0) "
      "Y0 Y1 Y2 Y3 + (-0.220041,0) Z1 + (0.165607,0) Z1 Z2 + (0.0454063,0) X0 "
      "X1 Y2 Y3 + (0.0454063,0) X0 X1 X2 X3";
  auto H = qcor::createObservable(opstr);
  // optimizer
  auto optimizer = qcor::createOptimizer("nlopt");

  // Create ADAPT-VQE instance
  // Run H2 with the singlet-adapted-uccsd pool
  qcor::ADAPT adapt(initial_state, H, 2, "singlet-adapted-uccsd", "vqe",
                    optimizer);

  // Execute!
  auto energy = adapt.execute();

  // Print the energy
  std::cout << std::setprecision(12) << energy << "\n";
}
