#include <iostream> 
#include <vector>
#include "qir_nisq_kernel_utils.hpp"
#include "qcor_qsim.hpp"

// Util pre-processor to wrap Q# operation 
// in a QCOR QuantumKernel.
// Compile:
// Note: need to use alpha package since this kernel will take a qubit array.
// qcor -qdk-version 0.17.2106148041-alpha deuteron.qs vqe_driver.cpp 
// Note: the first qreg argument is implicit.
qcor_import_qsharp_kernel(QCOR__ansatz, double);

int main() {
  // Allocate 2 qubits
  auto q = qalloc(2);
  // Hamiltonian
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);

  auto problemModel = QuaSiMo::ModelFactory::createModel(QCOR__ansatz, H, 2, 1);
  auto optimizer = createOptimizer("nlopt");
  // Instantiate a VQE workflow with the nlopt optimizer
  auto workflow = QuaSiMo::getWorkflow("vqe", {{"optimizer", optimizer}});

  // Result should contain the ground-state energy along with the optimal
  // parameters.
  auto result = workflow->execute(problemModel);

  const auto energy = result.get<double>("energy");
  std::cout << "Ground-state energy = " << energy << "\n";
  return 0;
}