#pragma once
#include "qcor_qsim.hpp"

namespace qcor {
namespace QuaSiMo {
// Time-dependent evolution workflow which can handle
// time-dependent Hamiltonian operator.
class TimeDependentWorkflow : public QuantumSimulationWorkflow {
public:
  virtual bool initialize(const HeterogeneousMap &params) override;
  virtual QuantumSimulationResult
  execute(const QuantumSimulationModel &model) override;
  virtual const std::string name() const override { return "td-evolution"; }
  virtual const std::string description() const override { return ""; }

private:
  HeterogeneousMap config_params;
  double t_0;
  double t_final;
  double dt;
  TdObservable ham_func;
};
} // namespace QuaSiMo
} // namespace qcor
