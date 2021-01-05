#include "qcor_qsim.hpp"

// High-level usage of Model Builder for Heisenberg model.
// The Heisenberg Hamiltonian is parameterized using ArQTiC scheme.

// Compile and run with:
/// $ qcor -qpu qpp TdWorkflowHeisenbergModel.cpp
/// $ ./a.out
int main(int argc, char **argv) {
  using ModelType = qcor::qsim::ModelFactory::ModelType;

  // Example ArQTiC input:
  // *Jz 
  // 0.01183898
  // *h_ext
  // 0.01183898
  // *initial_spins 
  // 0 0 0
  // *freq
  // 0.0048
  // *ext_dir
  // X
  // *num_spins
  // 3
  auto problemModel = qsim::ModelFactory::createModel(ModelType::Heisenberg,
                                                      {{"Jz", 0.01183898},
                                                       {"h_ext", 0.01183898},
                                                       {"freq", 0.0048},
                                                       {"ext_dir", "X"},
                                                       {"num_spins", 3}});
  // Workflow parameters:
  // *delta_t
  // 3
  // *steps
  // 20
  auto workflow = qsim::getWorkflow(
      "td-evolution", {{"dt", 3.0}, {"steps", 20}});

  // Result should contain the observable expectation value along Trotter steps.
  auto result = workflow->execute(problemModel);
  // Get the observable values (average magnetization)
  const auto obsVals = result.get<std::vector<double>>("exp-vals");

  // Print out for debugging:
  for (const auto &val : obsVals) {
    std::cout << "<Magnetization> = " << val << "\n";
  }

  return 0;
}
