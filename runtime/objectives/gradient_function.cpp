#include "gradient_function.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "qcor_utils.hpp"
#include "xacc_internal_compiler.hpp"
namespace qcor {
namespace __internal__ {
std::string DEFAULT_GRADIENT_METHOD = "central";

std::shared_ptr<GradientFunction>
get_gradient_method(const std::string &type,
                    std::shared_ptr<ObjectiveFunction> obj_func,
                    xacc::HeterogeneousMap options) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  auto service = xacc::getService<KernelGradientService>(type);
  service->initialize(obj_func, std::move(options));
  return service;
}

std::shared_ptr<GradientFunction>
get_gradient_method(const std::string &type,
                    std::function<std::shared_ptr<CompositeInstruction>(
                        std::vector<double>)>
                        kernel_eval,
                    Operator &obs) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  auto service = xacc::getService<KernelGradientService>(type);
  service->initialize(kernel_eval, obs);
  return service;
}
} // namespace __internal__
} // namespace qcor