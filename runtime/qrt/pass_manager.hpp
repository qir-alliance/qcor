#pragma once
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

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
  PassManager(int level, const std::vector<int> &qubitMap = {}, const std::string &placementName = "");
  // Static helper to run an optimization pass
  static PassStat runPass(const std::string &passName, std::shared_ptr<xacc::CompositeInstruction> program);
  // Default placement strategy
  static constexpr const char *DEFAULT_PLACEMENT = "swap-shortest-path";
  // Apply placement
  void applyPlacement(std::shared_ptr<xacc::CompositeInstruction> program) const;

  // Optimizes the input program.
  // Returns the full statistics about all the passes that have been executed.
  std::vector<PassStat>
  optimize(std::shared_ptr<xacc::CompositeInstruction> program) const;
  // List of passes for level 1:
  // Ordered list of passes to be executed.
  // Can have duplicated entries (run multiple times).
  static const constexpr char *const LEVEL1_PASSES[] = {
    "rotation-folding",
    // Merge single-qubit gates before running the circuit-optimizer
    // so that there are potentially more patterns emerged.
    "single-qubit-gate-merging",
    "circuit-optimizer",
  };

  // Level 2 is experimental, brute-force optimization
  // which could result in long runtime.
  static const constexpr char *const LEVEL2_PASSES[] = {
    "rotation-folding",
    "single-qubit-gate-merging",
    "circuit-optimizer",
    // Try to look for any two-qubit blocks
    // which can be simplified.
    "two-qubit-block-merging",
    // Re-run those simpler optimizers to 
    // make sure all simplification paterns are captured.
    "single-qubit-gate-merging",
    "circuit-optimizer",
  };
private:
  // Circuit optimization level
  int m_level;
  // Placement config.
  std::vector<int> m_qubitMap;
  std::string m_placement;
};
} // namespace internal
} // namespace qcor
