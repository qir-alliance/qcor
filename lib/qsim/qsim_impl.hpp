#pragma once
#include "qsim_interfaces.hpp"

namespace qcor {
// ========== Prototype Impl. ========================
// Example implementation:
class TrotterEvolution : public AnsatzGenerator {
public:
  Ansatz create_ansatz(Observable *obs,
                       const HeterogeneousMap &params) override;
};

// Estimate the cost function based on bitstring distribution,
// e.g. actual quantum hardware.
// Note: we can sub-class CostFunctionEvaluator to add post-processing or
// analysis of the result.
class BitCountExpectationEstimator : public CostFunctionEvaluator {
public:
  // Evaluate the cost
  virtual double
  evaluate(std::shared_ptr<CompositeInstruction> state_prep) override;

private:
  size_t nb_samples;
};

// VQE-type workflow which involves an optimization loop, i.e. an Optimizer.
class VqeWorkflow : public QsimWorkflow {
public:
  virtual bool initialize(const HeterogeneousMap &params) override;
  virtual QsimResult execute(const QsimModel &model) override;

  virtual const std::string name() const override { return "vqe"; }
  virtual const std::string description() const override { return ""; }
private:
  Optimizer *optimizer;
};



// Time-dependent evolution workflow which can handle
// time-dependent Hamiltonian operator.
class TimeDependentWorkflow : public QsimWorkflow {
public:
  virtual bool initialize(const HeterogeneousMap &params) override;
  virtual QsimResult execute(const QsimModel &model) override;
  virtual const std::string name() const override { return "td-evolution"; }
  virtual const std::string description() const override { return ""; }
private:
  static inline TimeDependentWorkflow *instance = nullptr;
  double t_0;
  double t_final;
  double dt;
  TdObservable ham_func;
};
} // namespace qcor