#pragma once
#include "qcor_qsim.hpp"

namespace qcor {
namespace qsim {
class PartialTomoObjFuncEval : public CostFunctionEvaluator {
public:
  // Evaluate the cost
  virtual double
  evaluate(std::shared_ptr<CompositeInstruction> state_prep) override;
  virtual const std::string name() const override { return "default"; }
  virtual const std::string description() const override { return ""; }
};

} // namespace qsim
} // namespace qcor