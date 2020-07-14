#pragma once
#include <unordered_map>
#include <vector>
#include <memory>

namespace xacc {
class CompositeInstruction;
}
namespace qcor {
namespace internal {
// Stats about an optimization pass:
struct PassStat
{
    // Name of the pass
    std::string passName;
    // Count per gate
    std::unordered_map<std::string, int> gateCountBefore;
    std::unordered_map<std::string, int> gateCountAfter;
    // Elapsed-time of this pass.
    double wallTimeMs;
};

class PassManager
{
public:
    PassManager(int level);
    // Optimizes the input program. 
    // Returns the full statistics about all the passes that have been executed.
    std::vector<PassStat> optimize(std::shared_ptr<xacc::CompositeInstruction> program) const;
    // List of passes for level 1:
    // Ordered list of passes to be executed. 
    // Can have duplicated entries (run multiple times).
    static const constexpr char* const LEVEL1_PASSES [] = { 
        "rotation-folding", 
        "circuit-optimizer",
        "swap-shortest-path"
    };
    // TODO: define other levels if neccesary:
    // e.g. could be looping those passes multiple times.
private: 
    int m_level;
};
}
}