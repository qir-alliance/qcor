#include "qcor_optimizer.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

namespace qcor {

std::shared_ptr<xacc::Optimizer> createOptimizer(const std::string &type,
                                                 HeterogeneousMap &&options) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::getOptimizer(type, options);
}

} // namespace qcor