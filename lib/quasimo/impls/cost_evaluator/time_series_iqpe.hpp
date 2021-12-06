/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#pragma once
#include "qcor_qsim.hpp"

namespace qcor {
namespace QuaSiMo {
// Evaluate the objective function based on QPE protocol.
class PhaseEstimationObjFuncEval : public CostFunctionEvaluator {
public:
  // Evaluate the cost
  virtual double
  evaluate(std::shared_ptr<CompositeInstruction> state_prep) override;
  virtual const std::string name() const override { return "qpe"; }
  virtual const std::string description() const override { return ""; }
};
} // namespace QuaSiMo
} // namespace qcor