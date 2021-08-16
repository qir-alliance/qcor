#include "ResourcesTracerAccelerator.hpp"
#include <iomanip>
#include <random>
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
  if (gateNameToCount.find(inst->name()) == gateNameToCount.end()) {
    gateNameToCount[inst->name()] = 1;
  } else {
    gateNameToCount[inst->name()] += 1;
  }
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
  // Print resources estimation result:
  // Currently, simply print gate count:
  std::cout << "Resources Estimation Result:\n";
  std::cout << "Number of qubit required: " << qubit_idxs.size() << "\n";
  std::cout << "Gate Count Report: \n";
  size_t totalNumberGates = 0;
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

  for (const auto &[gateName, count] : gateNameToCount) {
    printEachRow(gateName, count);
  }
  stream << std::string(totalWidth, '-') << "\n";
  std::cout << stream.str();
}
} // namespace qcor
