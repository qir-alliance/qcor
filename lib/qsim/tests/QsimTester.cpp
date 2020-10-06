#include "qcor_qsim.hpp"

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

  // Example: build model for Fig. 2 of
  // https://journals.aps.org/prb/pdf/10.1103/PhysRevB.101.184305
  auto problemModel = ModelBuilder::createModel({{"protocol", "time-evolution"},
                                                 {"method", "trotter"},
                                                 {"dt", 3.0},
                                                 {"steps", 100},
                                                 {"hamiltonian", H},
                                                 {"observable", observable}});

  auto workflow = problemModel.get_workflow();

  // Result should contain the observable expectation value along Trotter steps.
  auto result = workflow->execute();

  return 0;
}
