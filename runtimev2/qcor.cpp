#include "qcor.hpp"

#include "Optimizer.hpp"

#include "xacc.hpp"
#include "xacc_quantum_gate_api.hpp"
#include "xacc_service.hpp"

#include "qalloc.hpp"
#include "qrt.hpp"

namespace qcor {


namespace __internal__ {


std::shared_ptr<xacc::IRTransformation>
get_transformation(const std::string &transform_type) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::getService<xacc::IRTransformation>(transform_type);
}

} // namespace __internal__



} // namespace qcor