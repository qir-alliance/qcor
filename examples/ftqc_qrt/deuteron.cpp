#include "ftqc/vqe.hpp"
#include "xacc.hpp"
// Compile with:
// qcor -qpu qpp -qrt ftqc -I<qcor/examples/shared> deuteron.cpp 

__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}

int main(int argc, char **argv) {
  // Allocate 2 qubits
  auto q = qalloc(2);
  // Hamiltonian
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);

  const auto angles = xacc::linspace(-xacc::constants::pi, xacc::constants::pi, 20);
  for (const auto& angle: angles)
  {
    // Ansatz at a specific angle
    auto statePrep =
        std::function<void(qreg)>{[angle](qreg q) { ansatz(q, angle); }};
    double energy = 0.0;
    EstimateEnergy(q, statePrep, H, 1024, energy);
    std::cout << "Energy(" << angle << ") = " << energy << "\n";
  }
}
