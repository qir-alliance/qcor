#include "qcor_qsim.hpp"

__qpu__ void state_prep(qreg q) { X(q[0]); }

int main(int argc, char** argv) {
  set_verbose(true);
  using namespace QuaSiMo;

  // Create the qsearch IR Transformation
  auto qsearch_optimizer = createTransformation("qsearch");

  // Create the Hamiltonian
  auto observable = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) +
                    .21829 * Z(0) - 6.125 * Z(1) + 5.907;

  // We'll run with 3 steps and .1 step size
  const int steps = 3;
  const double stepSize = 0.1;

  // Create the model (2 qubits, 0 variational params in above state_prep)
  // and QITE workflow
  auto problemModel = ModelFactory::createModel(state_prep, &observable, observable.nBits(), 0);
  auto workflow =
      getWorkflow("qite", {{"steps", steps},
                           {"step-size", stepSize},
                           {"circuit-optimizer", qsearch_optimizer}});

  // Execute
  auto result = workflow->execute(problemModel);

  // Get the energy and final circuit
  const auto energy = result.get<double>("energy");
  auto finalCircuit = result.getPointerLike<CompositeInstruction>("circuit");
  printf("\n%s\nEnergy=%f\n", finalCircuit->toString().c_str(), energy);
}