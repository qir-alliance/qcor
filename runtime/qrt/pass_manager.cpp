#include "pass_manager.hpp"
#include "InstructionIterator.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
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
PassManager::PassManager(int level) : m_level(level) {}

std::vector<PassStat> PassManager::optimize(
    std::shared_ptr<xacc::CompositeInstruction> program) const {

  // We only support level 1 atm.
  if (m_level != 1) {
    return {};
  }

  std::vector<PassStat> passData;
  for (const auto &passName : LEVEL1_PASSES) {
    PassStat stat;
    stat.passName = passName;
    // Counts gate before:
    stat.gateCountBefore = PassStat::countGates(program);
    xacc::ScopeTimer timer(passName, false);
    auto xaccOptTransform = xacc::getService<xacc::IRTransformation>(passName);
    xaccOptTransform->apply(program, nullptr);
    // Stores the elapsed time.
    stat.wallTimeMs = timer.getDurationMs();
    // Counts gate after:
    stat.gateCountAfter = PassStat::countGates(program);
    passData.emplace_back(std::move(stat));
  }

  return passData;
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