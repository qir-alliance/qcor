#pragma once
#include "qcor_qsim.hpp"

namespace qcor {
namespace qsim {
class QiteWorkflow : public QuantumSimulationWorkflow {
public:
  virtual bool initialize(const HeterogeneousMap &params) override;
  virtual QuantumSimulationResult
  execute(const QuantumSimulationModel &model) override;

  virtual const std::string name() const override { return "qite"; }
  virtual const std::string description() const override { return ""; }

private:
  HeterogeneousMap config_params;
};
} // namespace qsim
} // namespace qcor