#include "ResourcesTracerAccelerator.hpp"

#include <iomanip>
#include <random>

#include "xacc_service.hpp"

using namespace xacc;
namespace qcor {
void TracerAccelerator::initialize(const HeterogeneousMap &params) {
  qubitIdToMeasureProbs.clear();

  if (params.keyExists<std::vector<double>>("meas1-probs")) {
    const auto meas1Probs = params.get<std::vector<double>>("meas1-probs");
    for (size_t i = 0; i < meas1Probs.size(); ++i) {
      if (meas1Probs[i] < 0.0 || meas1Probs[i] > 1.0) {
        xacc::error("Invalid measure probability setting: " +
                    std::to_string(meas1Probs[i]));
      }
      qubitIdToMeasureProbs[i] = meas1Probs[i];
    }
  }

  // Check for clifford+t == true, 1, or "true"
  use_clifford_t = params.get_or_default("clifford+t", false);
  if (!use_clifford_t)
    use_clifford_t = params.get_or_default("clifford+t", (int)0);
  if (!use_clifford_t)
    use_clifford_t =
        params.get_or_default("clifford+t", std::string("false")) == "true";

  counter_composite =
      xacc::getIRProvider("quantum")->createComposite("counter_composite");
}

// For NISQ: resource estimation is simply printing out the circuit....
void TracerAccelerator::execute(
    std::shared_ptr<AcceleratorBuffer> buffer,
    const std::shared_ptr<CompositeInstruction> compositeInstruction) {
  xacc::error("Unsupported!!!");
}
void TracerAccelerator::execute(
    std::shared_ptr<AcceleratorBuffer> buffer,
    const std::vector<std::shared_ptr<CompositeInstruction>>
        compositeInstructions) {
  xacc::error("Unsupported!!!");
}
void TracerAccelerator::apply(std::shared_ptr<AcceleratorBuffer> buffer,
                              std::shared_ptr<Instruction> inst) {
  for (const auto &bitId : inst->bits()) {
    qubit_idxs.emplace(bitId);
  }
  counter_composite->addInstruction(inst);
  // Emulate measure:
  if (inst->name() == "Measure") {
    const double meas1Prob = getMeas1Prob(inst->bits()[0]);
    static const auto generateRan = []() {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0, 1);
      return dis(gen);
    };

    // meas1Prob = 1.0 => always returns true
    // meas1Prob = 0.0 => always returns false
    const auto measRes = generateRan() < meas1Prob;
    buffer->measure(inst->bits()[0], (measRes ? 1 : 0));
  }
}

void TracerAccelerator::printResourcesEstimationReport() {
  if (use_clifford_t) {
    if (!xacc::hasService<xacc::IRTransformation>("gridsynth")) {
      xacc::error(
          "Cannot output clifford+t resources, gridsynth not "
          "installed. Install with \nqcor -install-plugin "
          "https://code.ornl.gov/qci/gridsynth");
    }
    auto irt = xacc::getIRTransformation("gridsynth");
    irt->apply(counter_composite, nullptr);
  }

  for (auto inst : counter_composite->getInstructions()) {
    if (gateNameToCount.find(inst->name()) == gateNameToCount.end()) {
      gateNameToCount[inst->name()] = 1;
    } else {
      gateNameToCount[inst->name()] += 1;
    }
  }

  // Print resources estimation result:
  // Currently, simply print gate count:
  std::cout << "Resources Estimation Result:\n";
  std::cout << "Number of qubits required: " << qubit_idxs.size() << "\n";
  size_t totalNumberGates = 0, totalCtrlOperations = 0;
  std::stringstream stream;
  const size_t nbColumns = 2;
  const size_t columnWidth = 8;
  const auto totalWidth = nbColumns * columnWidth + 6;
  stream << std::string(totalWidth, '-') << "\n";
  stream << "| " << std::left << std::setw(8) << "GATE"
         << " |";
  stream << std::left << std::setw(8) << "COUNT"

         << " |\n";
  stream << std::string(totalWidth, '-') << "\n";

  const auto printEachRow = [&](const std::string &gateName, int count) {
    stream << "| " << std::setw(8) << gateName << " |";
    stream << std::setw(8) << count << " |\n";
  };

  auto insts = xacc::getServices<xacc::Instruction>();
  std::vector<std::string> two_qubit_names;
  for (auto inst : insts) {
    if (inst->nRequiredBits() > 1) {
      two_qubit_names.push_back(inst->name());
    }
  }

  // Print each row, and count total number of instructions, and
  // count any 2 qubit control operations.
  for (const auto &[gateName, count] : gateNameToCount) {
    printEachRow(gateName, count);
    totalNumberGates += count;
    if (xacc::container::contains(two_qubit_names, gateName)) {
      totalCtrlOperations += count;
    }
  }

  stream << std::string(totalWidth, '-') << "\n";
  std::cout << "Total Gates: " << totalNumberGates << "\n";
  std::cout << "Total Control Operations: " << totalCtrlOperations << "\n";
  std::cout << "Gate Count Report: \n";
  std::cout << stream.str();
}
}  // namespace qcor
