#include "qcor_hybrid.hpp"
#include <iomanip>

// Define the state preparation kernel
__qpu__ void state_prep(qreg q) { X(q[0]); }

int main() {

  // Define the Hamiltonian using the QCOR API
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);

  // Create a QITE instance, give it
  // the parameterized state_prep functor,
  // Observable, and n_steps and step_size
  QITE qite(state_prep, H, 5, .1);

  // Execute!
  auto energy = qite.execute();

  // Print the energy
  std::cout << std::setprecision(12) << energy << "\n";
}
