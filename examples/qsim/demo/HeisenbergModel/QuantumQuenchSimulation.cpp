#include "qcor_qsim.hpp"

// Simulate dynamics of a quantum quench of
// a 1D antiferromagnetic Heisenberg model.

// Compile and run with:
/// $ qcor -qpu qsim QuantumQuenchSimulation.cpp
/// $ ./a.out
int main(int argc, char **argv) {
  // Handle input parameters:
  // --n-spins : number of spins/qubits (default = 7)
  // --g: g parameter - ZZ coupling strength (default = 0.0)
  // --dt: Trotter dt (default = 0.05)
  // --steps: number of steps (defalt = 100)

  // Process the input arguments
  std::vector<std::string> arguments(argv + 1, argv + argc);
  int n_spins = 7;
  double g = 0.0;
  double dt = 0.05;
  int n_steps = 100;

  for (int i = 0; i < arguments.size(); i++) {
    if (arguments[i] == "--n-spins") {
      n_spins = std::stoi(arguments[i + 1]);
    }
    if (arguments[i] == "--g") {
      g = std::stod(arguments[i + 1]);
    }
    if (arguments[i] == "--dt") {
      dt = std::stod(arguments[i + 1]);
    }
    if (arguments[i] == "--steps") {
      n_steps = std::stoi(arguments[i + 1]);
    }
  }

  std::cout << "Running QSim for a quenching model with " << n_spins
            << " spins and g = " << g << "\n";
  std::cout << "dt " << dt << "; number of steps = " << n_steps << "\n";
  using ModelType = qcor::qsim::ModelBuilder::ModelType;
  // Initial spin state: Neel state
  std::vector<int> initial_spins;
  for (int i = 0; i < n_spins; ++i) {
    // Alternating up/down spins
    initial_spins.emplace_back(i % 2);
  }

  auto problemModel = qsim::ModelBuilder::createModel(
      ModelType::Heisenberg, {{"Jx", 1.0},
                              {"Jy", 1.0},
                              // Jz == g parameter
                              {"Jz", g},
                              // No external field
                              {"h_ext", 0.0},
                              {"ext_dir", "X"},
                              {"num_spins", n_spins},
                              {"initial_spins", initial_spins},
                              {"observable", "staggered_magnetization"}});
  // Workflow parameters:
  auto workflow =
      qsim::getWorkflow("td-evolution", {{"dt", dt}, {"steps", n_steps}});

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
