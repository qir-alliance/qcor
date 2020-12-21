#include "qcor_qsim.hpp"

// Simulate dynamics of a quantum quench of 
// a 1D antiferromagnetic Heisenberg model.

// Compile and run with:
/// $ qcor -qpu qsim QuantumQuenchSimulation.cpp
/// $ ./a.out
int main(int argc, char **argv) {
  using ModelType = qcor::qsim::ModelBuilder::ModelType;
  // Initial spin state: Neel state (7 qubits/spins)
  const std::vector<int> initial_spins{0, 1, 0, 1, 0, 1, 0};
  auto problemModel = qsim::ModelBuilder::createModel(
      ModelType::Heisenberg, {{"Jx", 1.0},
                              {"Jy", 1.0},
                              {"Jz", 0.2},
                              {"h_ext", 0.0},
                              {"ext_dir", "X"},
                              {"num_spins", 7},
                              {"initial_spins", initial_spins},
                              {"observable", "staggered_magnetization"}});
  // Workflow parameters:
  auto workflow = qsim::getWorkflow(
      "td-evolution", {{"dt", 0.05}, {"steps", 100}});

  // Result should contain the observable expectation value along Trotter steps.
  auto result = workflow->execute(problemModel);
  // Get the observable values (average magnetization)
  const auto obsVals = result.get<std::vector<double>>("exp-vals");

  // Print out for debugging:
  std::cout << "<Staggered Magnetization> = \n"; 
  for (const auto &val : obsVals) {
    std::cout << val << "\n";
  }

  return 0;
}
