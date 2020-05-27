// In this demo, we use a built-in QAOA ansatz circuit
// with QCOR's VQE Objective Function.
// Note: we can also explicitly construct this ansatz circuit in QCOR.
// e.g., see qaoa_example.cpp
#include <qalloc>
#include "qcor.hpp"

__qpu__ void qaoa_ansatz(qreg q, int n, std::vector<double> betas, std::vector<double> gammas, std::shared_ptr<xacc::Observable> costHamiltonian, std::shared_ptr<xacc::Observable> refHamiltonian) {
  // Just use the built-in qaoa circuit
  qaoa(q, n, betas, gammas, costHamiltonian, refHamiltonian);
}

// Compile (using Qpp simulator backend)
// qcor -o qaoa-example -qpu qpp qaoa-builtin-circuit.cpp
int main(int argc, char **argv) {
  auto buffer = qalloc(2);
  auto optimizer = qcor::createOptimizer("nlopt");
  // Cost Hamiltonian
  auto costHam = 5.907-2.1433*qcor::X(0)*qcor::X(1)-2.1433*qcor::Y(0)*qcor::Y(1)+0.21829*qcor::Z(0)-6.125*qcor::Z(1);
  std::shared_ptr<qcor::Observable> observable = std::make_shared<qcor::PauliOperator>(costHam);
  // Mixer Hamiltonian
  auto refH = qcor::X(0) + qcor::X(1);
  std::shared_ptr<qcor::Observable> refHamiltonian = std::make_shared<qcor::PauliOperator>(refH);
  
  // VQE objective function
  auto vqe = qcor::createObjectiveFunction("vqe", qaoa_ansatz, observable);
  vqe->set_qreg(buffer);

  // QAOA variational parameters
  const int nbSteps = 2;
  const int nbParamsPerStep = 2 /*beta (mixer)*/ + 4 /*gamma (cost)*/;
  const int totalParams = nbSteps * nbParamsPerStep;
  int iterCount = 0;
  // Optimization function
  qcor::OptFunction f(
      [&](const std::vector<double> x, std::vector<double> &grad) {
        std::vector<double> betas;
        std::vector<double> gammas;
        // Unpack nlopt params
        // Beta: nbSteps * number qubits
        for (int i = 0; i < nbSteps * buffer.size(); ++i) {
          betas.emplace_back(x[i]);
        }

        for (int i = betas.size(); i < x.size(); ++i) {
          gammas.emplace_back(x[i]);
        }
        // Evaluate the objective function
        const double costVal = (*vqe)(buffer, buffer.size(), betas, gammas, observable, refHamiltonian);
        std::cout << "Iter " << iterCount << ": Cost = " << costVal << "\n";
        iterCount++;
        return costVal;
      },
      totalParams);
  auto results = optimizer->optimize(f);
  std::cout << "Final cost: " << results.first << "\n";
}
