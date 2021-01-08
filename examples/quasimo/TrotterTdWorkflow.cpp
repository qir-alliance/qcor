#include "qcor_qsim.hpp"

// High-level usage of Model Builder for time-dependent quantum simulation
// problem using Trotter method.

// Compile and run with:
/// $ qcor -qpu qpp TrotterTdWorkflow.cpp
/// $ ./a.out

int main(int argc, char **argv) {
  // Define the time-dependent Hamiltonian and observable operator using the
  // QCOR API Time-dependent Hamiltonian
  QuaSiMo::TdObservable H = [](double t) {
    // Parameters:
    const double Jz = 2 * M_PI * 2.86265 * 1e-3;
    const double epsilon = Jz; // Values: 0.2Jz, 0.5Jz, Jz, 5Jz
    const double omega = 4.8 * 2 * M_PI * 1e-3;
    return -Jz * Z(0) * Z(1) - Jz * Z(1) * Z(2) -
           epsilon * std::cos(omega * t) * (X(0) + X(1) + X(2));
  };

  // Observable = average magnetization
  auto observable = (1.0 / 3.0) * (Z(0) + Z(1) + Z(2));

  // Example: build model and TD workflow for Fig. 2 of
  // https://journals.aps.org/prb/pdf/10.1103/PhysRevB.101.184305
  auto problemModel = QuaSiMo::ModelFactory::createModel(observable, H);
  // Trotter step = 3fs, number of steps = 100 -> end time = 300fs
  auto workflow = QuaSiMo::getWorkflow(
      "td-evolution", {{"dt", 3.0}, {"steps", 100}});

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
