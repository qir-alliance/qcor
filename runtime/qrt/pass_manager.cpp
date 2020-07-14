#include "pass_manager.hpp"

namespace qcor {
namespace internal {
PassManager::PassManager(int level):
    m_level(level) {}

std::vector<PassStat> PassManager::optimize(std::shared_ptr<xacc::CompositeInstruction> program) const {
    // TODO
    return {};
}
}
}