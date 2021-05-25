#include "gradient_function.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "qcor_utils.hpp"

namespace qcor {
namespace __internal__ {
std::shared_ptr<GradientFunction> get_gradient_method(
    const std::string &type, std::shared_ptr<ObjectiveFunction> obj_func,
    std::function<std::shared_ptr<xacc::CompositeInstruction>(std::vector<double>)>
        &kernel_eval) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  auto service = xacc::getService<KernelGradientService>(type);
  service->initialize(obj_func, kernel_eval);
  return service;
}
} // namespace __internal__
} // namespace qcor