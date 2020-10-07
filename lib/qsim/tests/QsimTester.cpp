#include "qsim_interfaces.hpp"
// TODO: fix so that this can be build w/ the QCOR compiler.
using namespace qcor;

// High-level usage of Model Builder
int main(int argc, char **argv) {
  // Define the Hamiltonian using the QCOR API
  // Note: we have support for high-level integration w/ Psi4, OpenFermion,
  // PySCF, etc.
  // Time-dependent Hamiltonian
  TdObservable H = [](double t) {
    // See Eq. (1): these are dummy params atm
    const double Jz = 1.0;
    const double epsilon = 1.0;
    const double omega = 1.0;
    return -Jz * Z(0) * Z(1) - Jz * Z(1) * Z(2) -
           epsilon * std::cos(omega * t) * (X(0) + X(1) + X(2));
  };

  // Observable = average magnetization
  auto observable = (1.0 / 3.0) * (Z(0) + Z(1) + Z(2));

  // Example: build model and TD workflow for Fig. 2 of
  // https://journals.aps.org/prb/pdf/10.1103/PhysRevB.101.184305
  auto problemModel = ModelBuilder::createModel(&observable, H);
  auto workflow = qcor::getWorkflow(
      "td-evolution", {{"method", "trotter"}, {"dt", 3.0}, {"steps", 100}});

  // Result should contain the observable expectation value along Trotter steps.
  auto result = workflow->execute(problemModel);

  return 0;
}
