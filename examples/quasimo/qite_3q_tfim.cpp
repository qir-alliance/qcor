#include "qcor_qsim.hpp"

int main(int argc, char** argv) {
  set_verbose(true);
  using namespace QuaSiMo;

  // Create the qsearch IR Transformation
  auto qsearch_optimizer = createTransformation("qsearch");

  // Create the Hamiltonian: 3-qubit TFIM
  auto observable = Z(0) * Z(1) + Z(1) * Z(2) + X(0) + X(1) + X(2);
  observable *= -1;
  // We'll run with 20 steps and .45 step size
  const int nbSteps = 20;
  const double stepSize = 0.45;

  auto problemModel = ModelFactory::createModel(&observable);
  // With circuit optimization (QSearch)
  // auto workflow =
  //     getWorkflow("qite", {{"steps", nbSteps},
  //                          {"step-size", stepSize},
  //                          {"circuit-optimizer", qsearch_optimizer}});
  
  // Without circuit optimization 
  auto workflow =
      getWorkflow("qite", {{"steps", nbSteps}, {"step-size", stepSize}});
  // Execute
  auto result = workflow->execute(problemModel);

  // Get the energy and final circuit
  const auto energy = result.get<double>("energy");
  auto finalCircuit = result.getPointerLike<CompositeInstruction>("circuit");
  printf("\n%s\nEnergy=%f\n", finalCircuit->toString().c_str(), energy);
  const auto energyAtStep = result.get<std::vector<double>>("exp-vals");
  std::cout << "QITE energy: [ ";
  for (const auto &val : energyAtStep) {
    std::cout << val << " ";
  }
  std::cout << "]\n";
}