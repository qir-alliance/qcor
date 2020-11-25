#pragma once
#include "qcor_qsim.hpp"

namespace qcor {
namespace qsim {
// Evaluate the objective function based on QPE protocol.
class PhaseEstimationObjFuncEval : public CostFunctionEvaluator {
public:
  // Evaluate the cost
  virtual double
  evaluate(std::shared_ptr<CompositeInstruction> state_prep) override;
  virtual const std::string name() const override { return "qpe"; }
  virtual const std::string description() const override { return ""; }
};
} // namespace qsim
} // namespace qcor