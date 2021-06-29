#include "iterative_qpe.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

namespace qcor {
namespace QuaSiMo {
bool IterativeQpeWorkflow::initialize(const HeterogeneousMap &params) {
  // Default params:
  num_steps = 1;
  num_iters = 1;
  if (params.keyExists<int>("time-steps")) {
    num_steps = params.get<int>("time-steps");
  }

  if (params.keyExists<int>("iterations")) {
    num_iters = params.get<int>("iterations");
  }

  // defaults to true all the way down, this really 
  // just gives us a way to turn it off
  cau_opt = params.get_or_default("cau-opt", true);

  return (num_steps >= 1) && (num_iters >= 1);
}

std::shared_ptr<CompositeInstruction>
IterativeQpeWorkflow::constructQpeTrotterCircuit(
    std::shared_ptr<Operator> op, double trotter_step, size_t nbQubits,
    double compensatedAncRot, int steps, int k, double omega, bool cau_opt) {
  auto provider = xacc::getIRProvider("quantum");
  auto kernel = provider->createComposite("__TEMP__QPE__LOOP__");
  // Ancilla qubit is the last one in the register.
  const size_t ancBit = nbQubits;
  std::shared_ptr<xacc::Observable> obs =
      std::dynamic_pointer_cast<xacc::Observable>(op->get_as_opaque());
  // Hadamard on ancilla qubit
  kernel->addInstruction(provider->createInstruction("H", ancBit));
  // Add a pre-compensated angle (for noise mitigation)
  if (std::abs(compensatedAncRot) > 1e-12) {
    kernel->addInstruction(
        provider->createInstruction("Rz", {ancBit}, {compensatedAncRot}));
  }
  // Using Trotter evolution method to generate U:
  // TODO: support other methods (e.g. Suzuki)
  auto method = xacc::getService<AnsatzGenerator>("trotter");
  auto trotterCir =
      method->create_ansatz(op.get(), {{"dt", trotter_step}, {"cau-opt", cau_opt}}).circuit;
  // std::cout << "Trotter circ:\n" << trotterCir->toString() << "\n";

  // Controlled-U
  auto ctrlKernel = std::dynamic_pointer_cast<xacc::CompositeInstruction>(
      xacc::getService<xacc::Instruction>("C-U"));
  ctrlKernel->expand({
      std::make_pair("U", trotterCir),
      std::make_pair("control-idx", static_cast<int>(ancBit)),
  });

  // Apply C-U^n
  int power = 1 << (k - 1);
  for (int i = 0; i < power * steps; ++i) {
    for (int instId = 0; instId < ctrlKernel->nInstructions(); ++instId) {
      // We need to clone the instruction since it'll be repeated.
      kernel->addInstruction(ctrlKernel->getInstruction(instId)->clone());
    }
  }

  // Rz on ancilla qubit
  // Global phase due to identity pauli
  if (obs->getIdentitySubTerm()) {
    const double idCoeff = obs->getIdentitySubTerm()->coefficient().real();
    const double globalPhase = 2 * M_PI * idCoeff * power;
    // std::cout << "Global phase = " << globalPhase << "\n";
    kernel->addInstruction(
        provider->createInstruction("Rz", {ancBit}, {globalPhase}));
  }

  kernel->addInstruction(provider->createInstruction("Rz", {ancBit}, {omega}));
  // Cancel the noise-mitigation angle:
  if (std::abs(compensatedAncRot) > 1e-12) {
    kernel->addInstruction(
        provider->createInstruction("Rz", {ancBit}, {-compensatedAncRot}));
  }
  return std::make_shared<CompositeInstruction>(kernel);
}

std::shared_ptr<CompositeInstruction> IterativeQpeWorkflow::constructQpeCircuit(
    std::shared_ptr<Operator> obs, int k, double omega, bool measure) const {
  auto provider = xacc::getIRProvider("quantum");
  const double trotterStepSize = -2 * M_PI / num_steps;
  auto kernel = constructQpeTrotterCircuit(obs, trotterStepSize, obs->nBits(),
                                           0.0, num_steps, k, omega, cau_opt);
  const auto nbQubits = obs->nBits();

  // Ancilla qubit is the last one in the register
  const size_t ancBit = nbQubits;
  // Hadamard on ancilla qubit (measure in X basis for regular IQPE)
  kernel->addInstruction(provider->createInstruction("H", ancBit));

  if (measure) {
    kernel->addInstruction(provider->createInstruction("Measure", ancBit));
  }

  return kernel;
}

void IterativeQpeWorkflow::HamOpConverter::fromObservable(Operator *obs) {
  translation = 0.0;
  for (auto &term : obs->getSubTerms()) {
    translation += std::abs(term.coefficient());
  }
  stretch = 0.5 / translation;
}

std::shared_ptr<Operator>
IterativeQpeWorkflow::HamOpConverter::stretchObservable(Operator *obs) const {
  Operator *pauliCast = obs;
  if (pauliCast) {
    auto result =
        std::make_shared<Operator>("pauli", std::to_string(translation) + " I");
    result->operator+=(*pauliCast);
    result->operator*=(stretch);
    return result;
  } else {
    return nullptr;
  }
}

double
IterativeQpeWorkflow::HamOpConverter::computeEnergy(double phaseVal) const {
  return phaseVal / stretch - translation;
}

QuantumSimulationResult
IterativeQpeWorkflow::execute(const QuantumSimulationModel &model) {
  ham_converter.fromObservable(model.observable);
  auto stretchedObs = ham_converter.stretchObservable(model.observable);
  // std::cout << "Stretched Obs: " << stretchedObs->toString() << "\n";
  auto provider = xacc::getIRProvider("quantum");
  // Iterative Quantum Phase Estimation:
  // We're using XACC IR construction API here, since using QCOR kernels here
  // seems to be complicated.
  double omega_coef = 0.0;
  std::vector<int> n_instructions;
  // Iterates over the num_iters
  // k runs from the number of iterations back to 1
  for (int iterIdx = 0; iterIdx < num_iters; ++iterIdx) {
    // State prep: evolves the qubit register to the initial quantum state, i.e.
    // the eigenvector state to estimate the eigenvalue.
    auto kernel = provider->createComposite("__TEMP__ITER_QPE__");
    if (model.user_defined_ansatz) {
      kernel->addInstruction(model.user_defined_ansatz->evaluate_kernel({})->as_xacc());
    }
    omega_coef = omega_coef / 2.0;
    // Construct the QPE circuit and append to the kernel:
    auto k = num_iters - iterIdx;

    auto iterQpe = constructQpeCircuit(stretchedObs, k, -2 * M_PI * omega_coef);
    kernel->addInstruction(iterQpe->as_xacc());
    int count = 0;
    xacc::InstructionIterator iter(kernel);
    while(iter.hasNext()) {
      auto next = iter.next();
      if (next->isEnabled() && !next->isComposite()) {
        count++;
      }
    }
    n_instructions.push_back(count);
    // Executes the iterative QPE algorithm:
    auto temp_buffer = xacc::qalloc(stretchedObs->nBits() + 1);
    // std::cout << "Kernel: \n" << kernel->toString() << "\n";
    xacc::internal_compiler::execute(temp_buffer.get(), kernel);
    // temp_buffer->print();

    // Estimate the phase value's bit at this iteration,
    // i.e. get the most-probable measure bit.
    const bool bitResult = [&temp_buffer]() {
      if (!temp_buffer->getMeasurementCounts().empty()) {
        // If the QPU returns bitstrings:
        if (xacc::container::contains(temp_buffer->getMeasurements(), "0")) {
          if (xacc::container::contains(temp_buffer->getMeasurements(), "1")) {
            return temp_buffer->computeMeasurementProbability("1") >
                   temp_buffer->computeMeasurementProbability("0");
          } else {
            return false;
          }
        } else {
          assert(
              xacc::container::contains(temp_buffer->getMeasurements(), "1"));
          return true;
        }
      } else {
        // If the QPU returns *expected* Z value:
        return temp_buffer->getExpectationValueZ() < 0.0;
      }
    }();

    if (bitResult) {
      omega_coef = omega_coef + 0.5;
    }
    // std::cout << "Iter " << iterIdx << ": Result = " << bitResult << ";
    // omega_coef = " << omega_coef << "\n";
  }

  return {{"phase", omega_coef}, {"n-kernel-instructions", n_instructions},
          {"energy", ham_converter.computeEnergy(omega_coef)}};
}
} // namespace QuaSiMo
} // namespace qcor