#include "qcor_hybrid.hpp"

using namespace qcor;

int main() {

  // Define the Hamiltonian using the QCOR API
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);

  // Define the reference hamiltonian
  auto ref_ham = X(0) + X(1);

  // Create a gradient based optimizer
  // If we don't provide this, QAOA will
  // use NLOpt QAOA. 
  auto lbfgs = qcor::createOptimizer(
      "nlopt", {std::make_pair("nlopt-optimizer", "l-bfgs")});

  // Turn on verbose output to see 
  // the iterations progress
  qcor::set_verbose(true);

  // we want 2 qaoa steps
  auto steps = 2;

  // Create the QAOA instance
  QAOA qaoa(H, ref_ham, steps);

  // Execute synchronously and display
  const auto [energy, params] = qaoa.execute(lbfgs);

  std::cout << "<H> = " << energy << "\n";

}