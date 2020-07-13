#include "operator_algebra.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "xacc_internal_compiler.hpp"


namespace qcor{
    PauliOperator transform(FermionOperator &obs, std::string transf) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  auto obsv = std::make_shared<FermionOperator>(obs);
  auto terms = std::dynamic_pointer_cast<PauliOperator>(
      xacc::getService<xacc::ObservableTransform>(transf)->transform(obsv));
  return *terms;
}
}


