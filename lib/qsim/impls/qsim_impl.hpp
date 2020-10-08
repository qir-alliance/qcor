#pragma once
#include "qcor_qsim.hpp"

namespace qcor {
// ========== Prototype Impl. ========================
// Example implementation:

// 1st-order Trotterization 
class TrotterEvolution : public AnsatzGenerator {
public:
  Ansatz create_ansatz(Observable *obs,
                       const HeterogeneousMap &params) override;
  virtual const std::string name() const override { return "trotter"; }
  virtual const std::string description() const override { return ""; }
};

// Unitary Coupled-Cluster Style ansatz construction
// TODO: not yet implemeneted.
class Ucc : public AnsatzGenerator {
public:
  Ansatz create_ansatz(Observable *obs,
                       const HeterogeneousMap &params) override;
  virtual const std::string name() const override { return "ucc"; }
  virtual const std::string description() const override { return ""; }
};

// other methods.


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
class VqeWorkflow : public QuatumSimulationWorkflow {
public:
  virtual bool initialize(const HeterogeneousMap &params) override;
  virtual QuatumSimulationResult execute(const QuatumSimulationModel &model) override;

  virtual const std::string name() const override { return "vqe"; }
  virtual const std::string description() const override { return ""; }
private:
  std::shared_ptr<Optimizer> optimizer;
};



// Time-dependent evolution workflow which can handle
// time-dependent Hamiltonian operator.
class TimeDependentWorkflow : public QuatumSimulationWorkflow {
public:
  virtual bool initialize(const HeterogeneousMap &params) override;
  virtual QuatumSimulationResult execute(const QuatumSimulationModel &model) override;
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