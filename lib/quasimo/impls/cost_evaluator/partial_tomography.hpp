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
#pragma once
#include "qcor_qsim.hpp"

namespace qcor {
namespace QuaSiMo {
class PartialTomoObjFuncEval : public CostFunctionEvaluator {
public:
  // Evaluate the cost
  virtual double
  evaluate(std::shared_ptr<CompositeInstruction> state_prep) override;
  virtual std::vector<double> evaluate(
      std::vector<std::shared_ptr<CompositeInstruction>> state_prep_circuits)
      override;
  virtual const std::string name() const override { return "default"; }
  virtual const std::string description() const override { return ""; }
};

} // namespace QuaSiMo
} // namespace qcor