#include "qcor_qsim.hpp"

// Demonstrate qsim's QITE workflow
// with a simple 1-qubit Hamiltonian from
// https://www.nature.com/articles/s41567-019-0704-4 Compile and run with:
/// $ qcor -qpu qpp QiteWorkflow.cpp
/// $ ./a.out

int main(int argc, char **argv) {
  auto ham = 0.7071067811865475 * X(0) + 0.7071067811865475 * Z(0);
  // Number of QITE time steps and step size
  const int nbSteps = 25;
  const double stepSize = 0.1;
  auto problemModel = qsim::ModelBuilder::createModel(ham);
  auto workflow =
      qsim::getWorkflow("qite", {{"steps", nbSteps}, {"step-size", stepSize}});
  auto result = workflow->execute(problemModel);
  const auto energy = result.get<double>("energy");
  const auto energyAtStep = result.get<std::vector<double>>("exp-vals");
  std::cout << "QITE energy: [ ";
  for (const auto &val : energyAtStep) {
    std::cout << val << " ";
  }
  std::cout << "]\n";
  std::cout << "Ground state energy: " << energy << "\n";
  return 0;
}