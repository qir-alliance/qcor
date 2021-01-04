#include "xacc.hpp"
#include "xacc_service.hpp"

// Include all the component implementations:
#include "ansatz_generator/trotter.hpp"
#include "cost_evaluator/partial_tomography.hpp"
#include "cost_evaluator/time_series_iqpe.hpp"
#include "workflow/iterative_qpe.hpp"
#include "workflow/qaoa.hpp"
#include "workflow/qite.hpp"
#include "workflow/time_dependent.hpp"
#include "workflow/vqe.hpp"
#include "workflow/adapt.hpp"

namespace qcor {
namespace qsim {
std::shared_ptr<QuantumSimulationWorkflow>
getWorkflow(const std::string &name, const HeterogeneousMap &init_params) {
  auto qsim_workflow = xacc::getService<QuantumSimulationWorkflow>(name);
  if (qsim_workflow && qsim_workflow->initialize(init_params)) {
    return qsim_workflow;
  }
  // ERROR: unknown workflow or invalid initialization options.
  return nullptr;
}
} // namespace qsim
} // namespace qcor

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
namespace {
using namespace cppmicroservices;
class US_ABI_LOCAL QuantumSimulationActivator : public BundleActivator {

public:
  QuantumSimulationActivator() {}

  void Start(BundleContext context) {
    using namespace qcor;
    // Workflow:
    context.RegisterService<qsim::QuantumSimulationWorkflow>(
        std::make_shared<qsim::TimeDependentWorkflow>());
    context.RegisterService<qsim::QuantumSimulationWorkflow>(
        std::make_shared<qsim::VqeWorkflow>());
    context.RegisterService<qsim::QuantumSimulationWorkflow>(
        std::make_shared<qsim::QaoaWorkflow>());
    context.RegisterService<qsim::QuantumSimulationWorkflow>(
        std::make_shared<qsim::QiteWorkflow>());
    context.RegisterService<qsim::QuantumSimulationWorkflow>(
        std::make_shared<qsim::IterativeQpeWorkflow>());
    context.RegisterService<qsim::QuantumSimulationWorkflow>(
        std::make_shared<qsim::AdaptVqeWorkflow>());
    // Cost evaluators
    context.RegisterService<qsim::CostFunctionEvaluator>(
        std::make_shared<qsim::PartialTomoObjFuncEval>());
    context.RegisterService<qsim::CostFunctionEvaluator>(
        std::make_shared<qsim::PhaseEstimationObjFuncEval>());

    // Ansatz generator
    context.RegisterService<qsim::AnsatzGenerator>(
        std::make_shared<qsim::TrotterEvolution>());
  }

  void Stop(BundleContext) {}
};
} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(QuantumSimulationActivator)