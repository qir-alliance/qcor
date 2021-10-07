/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
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
namespace QuaSiMo {
std::shared_ptr<QuantumSimulationWorkflow>
getWorkflow(const std::string &name, const HeterogeneousMap &init_params) {
  auto qsim_workflow = xacc::getService<QuantumSimulationWorkflow>(name);
  if (qsim_workflow && qsim_workflow->initialize(init_params)) {
    return qsim_workflow;
  }
  // ERROR: unknown workflow or invalid initialization options.
  return nullptr;
}
} // namespace QuaSiMo
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
    context.RegisterService<QuaSiMo::QuantumSimulationWorkflow>(
        std::make_shared<QuaSiMo::TimeDependentWorkflow>());
    context.RegisterService<QuaSiMo::QuantumSimulationWorkflow>(
        std::make_shared<QuaSiMo::VqeWorkflow>());
    context.RegisterService<QuaSiMo::QuantumSimulationWorkflow>(
        std::make_shared<QuaSiMo::QaoaWorkflow>());
    context.RegisterService<QuaSiMo::QuantumSimulationWorkflow>(
        std::make_shared<QuaSiMo::QiteWorkflow>());
    context.RegisterService<QuaSiMo::QuantumSimulationWorkflow>(
        std::make_shared<QuaSiMo::IterativeQpeWorkflow>());
    context.RegisterService<QuaSiMo::QuantumSimulationWorkflow>(
        std::make_shared<QuaSiMo::AdaptVqeWorkflow>());
    // Cost evaluators
    context.RegisterService<QuaSiMo::CostFunctionEvaluator>(
        std::make_shared<QuaSiMo::PartialTomoObjFuncEval>());
    context.RegisterService<QuaSiMo::CostFunctionEvaluator>(
        std::make_shared<QuaSiMo::PhaseEstimationObjFuncEval>());

    // Ansatz generator
    context.RegisterService<QuaSiMo::AnsatzGenerator>(
        std::make_shared<QuaSiMo::TrotterEvolution>());
  }

  void Stop(BundleContext) {}
};
} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(QuantumSimulationActivator)