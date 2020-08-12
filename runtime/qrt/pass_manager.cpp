#include "pass_manager.hpp"
#include "InstructionIterator.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "xacc_internal_compiler.hpp"
#include <iomanip>
#include <numeric>
namespace {
std::string
printGateCountComparison(const std::unordered_map<std::string, int> &before,
                         const std::unordered_map<std::string, int> &after) {
  std::stringstream stream;
  const size_t nbColumns = 3;
  const size_t columnWidth = 8;
  const auto totalWidth = nbColumns * columnWidth + 8;
  stream << std::string(totalWidth, '-') << "\n";
  // Print headers:
  stream << "| " << std::left << std::setw(8) << "GATE"
         << " |";
  stream << std::left << std::setw(8) << "BEFORE"
         << " |";
  stream << std::left << std::setw(8) << "AFTER"
         << " |\n";
  stream << std::string(totalWidth, '-') << "\n";

  const auto printEachRow = [&](const std::string &gateName, int countBefore,
                                int countAfter) {
    stream << "| " << std::setw(8) << gateName << " |";
    stream << std::setw(8) << countBefore << " |";
    stream << std::setw(8) << countAfter << " |\n";
  };

  for (const auto &[gateName, countBefore] : before) {
    const auto iter = after.find(gateName);
    const auto countAfter = (iter == after.end()) ? 0 : iter->second;
    printEachRow(gateName, countBefore, countAfter);
  }
  stream << std::string(totalWidth, '-') << "\n";
  return stream.str();
}
} // namespace

namespace qcor {
namespace internal {
PassManager::PassManager(int level, const std::vector<int> &qubitMap,
                         const std::string &placementName)
  : m_level(level), m_qubitMap(qubitMap), m_placement(placementName) {}

PassStat PassManager::runPass(const std::string &passName, std::shared_ptr<xacc::CompositeInstruction> program) {
  PassStat stat;
  stat.passName = passName;
  // Counts gate before:
  stat.gateCountBefore = PassStat::countGates(program);
  xacc::ScopeTimer timer(passName, false);
  
  if (!xacc::hasService<xacc::IRTransformation>(passName)
      && !xacc::hasContributedService<xacc::IRTransformation>(passName)) {
    // Graciously ignores passes which cannot be located.
    // Returns empty stats
    return stat;
  }

  auto xaccOptTransform =
      xacc::getIRTransformation(passName);
  if (xaccOptTransform) {
    xaccOptTransform->apply(program, nullptr);
  }
  // Stores the elapsed time.
  stat.wallTimeMs = timer.getDurationMs();
  // Counts gate after:
  stat.gateCountAfter = PassStat::countGates(program);
  return stat;
}

std::vector<PassStat> PassManager::optimize(
    std::shared_ptr<xacc::CompositeInstruction> program) const {
  std::vector<PassStat> passData;
  // Selects the list of passes based on the optimization level.
  const auto passesToRun = [&]() {
    if (m_level == 1) {
      return std::vector<std::string>(std::begin(LEVEL1_PASSES),
                                      std::end(LEVEL1_PASSES));
    } else if (m_level == 2) {
      return std::vector<std::string>(std::begin(LEVEL2_PASSES),
                                      std::end(LEVEL2_PASSES));
    }
    return std::vector<std::string>();
  }();

  for (const auto &passName : passesToRun) {
    passData.emplace_back(runPass(passName, program));
  }

  return passData;
}

void PassManager::applyPlacement(std::shared_ptr<xacc::CompositeInstruction> program) const {
  const std::string placementName = [&]() -> std::string {
    // If the qubit-map was provided, always use default-placement
    if (!m_qubitMap.empty()) {
      return "default-placement";
    }
    // Use the specified placement if any.
    // Note: placement will only be activated if the accelerator
    // has a connectivity graph.
    return m_placement.empty() ? DEFAULT_PLACEMENT : m_placement;
  }();
  
  if (!xacc::hasService<xacc::IRTransformation>(placementName)
      && !xacc::hasContributedService<xacc::IRTransformation>(placementName)) {
    // Graciously ignores services which cannot be located.
    return;
  }

  auto irt = xacc::getIRTransformation(placementName);
  if (irt->type() == xacc::IRTransformationType::Placement &&
    xacc::internal_compiler::qpu &&
    !xacc::internal_compiler::qpu->getConnectivity().empty()) {
    if (placementName == "default-placement") {
      irt->apply(program, xacc::internal_compiler::qpu, {{"qubit-map", m_qubitMap}});
    } else {
      irt->apply(program, xacc::internal_compiler::qpu);
    }
  }
}

std::unordered_map<std::string, int> PassStat::countGates(
    const std::shared_ptr<xacc::CompositeInstruction> &program) {
  std::unordered_map<std::string, int> gateCount;
  xacc::InstructionIterator iter(program);
  while (iter.hasNext()) {
    auto next = iter.next();
    if (!next->isComposite()) {
      if (gateCount.find(next->name()) == gateCount.end()) {
        gateCount[next->name()] = 1;
      } else {
        gateCount[next->name()] += 1;
      }
    }
  }
  return gateCount;
}

std::string PassStat::toString(bool shortForm) const {
  const auto countNumberOfGates =
      [](const std::unordered_map<std::string, int> &gateCount) {
        return std::accumulate(gateCount.begin(), gateCount.end(), 0,
                               [](const int previousSum, const auto &element) {
                                 return previousSum + element.second;
                               });
      };

  std::stringstream ss;
  const std::string separator(40, '*');
  ss << separator << "\n";
  ss << std::string((separator.size() - passName.size()) / 2, ' ') << passName
     << "\n";
  ss << separator << "\n";
  ss << " - Elapsed time: " << wallTimeMs << " [ms]\n";
  ss << " - Number of Gates Before: " << countNumberOfGates(gateCountBefore)
     << "\n";
  ss << " - Number of Gates After: " << countNumberOfGates(gateCountAfter)
     << "\n";

  if (!shortForm) {
    // Prints the full gate count table if required (long form)
    ss << printGateCountComparison(gateCountBefore, gateCountAfter);
  }
  return ss.str();
}

} // namespace internal
} // namespace qcor