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
#include "objective_function.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "xacc_internal_compiler.hpp"
namespace qcor {
namespace __internal__ {
std::shared_ptr<ObjectiveFunction> get_objective(const std::string &type) {
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