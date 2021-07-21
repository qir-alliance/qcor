#include "objective_function.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "xacc_internal_compiler.hpp"
// for _QCOR_MUTEX
#include "qcor_config.hpp"

#ifdef _QCOR_MUTEX
#include <mutex>
#pragma message ("_QCOR_MUTEX is ON")
#endif

namespace qcor {
namespace __internal__ {

#ifdef _QCOR_MUTEX
std::mutex qcor_xacc_init_lock;
#endif

std::shared_ptr<ObjectiveFunction> get_objective(const std::string &type) {
#ifdef _QCOR_MUTEX
  std::lock_guard<std::mutex> lock(qcor_xacc_init_lock);
#endif
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::getService<ObjectiveFunction>(type);
}
} // namespace __internal__



// std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
//     const std::string &obj_name, std::shared_ptr<CompositeInstruction> kernel,
//     std::shared_ptr<Observable> observable, HeterogeneousMap &&options) {
//   auto obj_func = qcor::__internal__::get_objective(obj_name);
//   obj_func->initialize(observable.get(), kernel);
//   obj_func->set_options(options);
//   return obj_func;
// }

// std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
//     const std::string &obj_name, std::shared_ptr<CompositeInstruction> kernel,
//     Observable &observable, HeterogeneousMap &&options) {
//   auto obj_func = qcor::__internal__::get_objective(obj_name);
//   obj_func->initialize(&observable, kernel);
//   obj_func->set_options(options);
//   return obj_func;
// }
} // namespace qcor