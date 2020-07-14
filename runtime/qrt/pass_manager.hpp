#pragma once
#include <memory>
#include <unordered_map>
#include <vector>

namespace xacc {
class CompositeInstruction;
}
namespace qcor {
namespace internal {
// Stats about an optimization pass:
struct PassStat {
  // Name of the pass
  std::string passName;
  // Count per gate
  std::unordered_map<std::string, int> gateCountBefore;
  std::unordered_map<std::string, int> gateCountAfter;
  // Elapsed-time of this pass.
  double wallTimeMs;
  // Helper to collect stats.
  static std::unordered_map<std::string, int>
  countGates(const std::shared_ptr<xacc::CompositeInstruction> &program);
  // Pretty printer.
  std::string toString(bool shortForm = true) const;
};

class PassManager {
public:
  PassManager(int level);
  // Optimizes the input program.
  // Returns the full statistics about all the passes that have been executed.
  std::vector<PassStat>
  optimize(std::shared_ptr<xacc::CompositeInstruction> program) const;
  // List of passes for level 1:
  // Ordered list of passes to be executed.
  // Can have duplicated entries (run multiple times).
  static const constexpr char *const LEVEL1_PASSES[] = {"rotation-folding",
                                                        "circuit-optimizer"};
  // TODO: define other levels if neccesary:
  // e.g. could be looping those passes multiple times.
private:
  int m_level;
};
} // namespace internal
} // namespace qcor