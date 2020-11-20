#pragma once
#include "qcor_qsim.hpp"

namespace qcor {
namespace qsim {
// VQE-type workflow which involves an optimization loop, i.e. an Optimizer.
class VqeWorkflow : public QuantumSimulationWorkflow {
public:
  virtual bool initialize(const HeterogeneousMap &params) override;
  virtual QuantumSimulationResult
  execute(const QuantumSimulationModel &model) override;

  virtual const std::string name() const override { return "vqe"; }
  virtual const std::string description() const override { return ""; }

private:
  std::shared_ptr<Optimizer> optimizer;
  HeterogeneousMap config_params;
};
} // namespace qsim
} // namespace qcor