#include "qcor_qsim.hpp"

// Using noise-mitigation observable evaluation (Verified QPE)

// Compile and run with (e.g. using the 5-qubit Yorktown device)
/// $ qcor -qpu -qpu aer[noise-model:<noise JSON>] -shots 8192 -opt-pass
/// two-qubit-block-merging VerifiedQuantumPhaseEstimation.cpp
/// $ ./a.out

// For this simple circuit, the Trotter circuit can be significantly simplified
// by the `two-qubit-block-merging` optimizer (combine Trotter steps and
// decompose the total circuit into a more efficient KAK circuit).

// Note: for simplicity, we use the time-dependent Ising model with only 2
// qubits, hence with QPE, we need a total of 3 qubits. These 3 qubits can be
// directly mapped to the triangular topology of the Yorktown device.
// (Ref: Fig 3 of PhysRevB.101.184305)
int main(int argc, char **argv) {
  // Define the time-dependent Hamiltonian and observable operator using the
  // QCOR API Time-dependent Hamiltonian
  qsim::TdObservable H = [](double t) {
    // Parameters:
    const double Jz = 2 * M_PI * 2.86265 * 1e-3;
    const double epsilon = Jz;
    const double omega = 4.8 * 2 * M_PI * 1e-3;
    return -Jz * Z(0) * Z(1) - epsilon * std::cos(omega * t) * (X(0) + X(1));
  };

  // Observable = average magnetization
  auto observable = (1.0 / 2.0) * (Z(0) + Z(1));

  // Example: build model and TD workflow for Fig. 2 of
  // https://journals.aps.org/prb/pdf/10.1103/PhysRevB.101.184305
  auto problemModel = qsim::ModelBuilder::createModel(observable, H);
  // Trotter step = 3fs, number of steps = 100 -> end time = 300fs
  // Request the Verified QPE observable evaluator:
  auto vqpeEvaluator =
      qsim::getObjEvaluator(observable, "qpe", {{"verified", true}});
  auto workflow = qsim::getWorkflow(
      "td-evolution",
      {{"dt", 3.0}, {"steps", 100}, {"evaluator", vqpeEvaluator}});

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
