#pragma once
#include "qcor_qsim.hpp"

namespace qcor {
namespace QuaSiMo {
class QiteWorkflow : public QuantumSimulationWorkflow {
protected:
  std::vector<std::shared_ptr<xacc::IRTransformation>> extra_circuit_optimizers;
  
public:
  virtual bool initialize(const HeterogeneousMap &params) override;
  virtual QuantumSimulationResult
  execute(const QuantumSimulationModel &model) override;

  virtual const std::string name() const override { return "qite"; }
  virtual const std::string description() const override { return ""; }

private:
  HeterogeneousMap config_params;
};
} // namespace QuaSiMo
} // namespace qcor