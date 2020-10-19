#pragma once
#include "qcor_qsim.hpp"

namespace qcor {
namespace qsim {
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
  double t_0;
  double t_final;
  double dt;
  TdObservable ham_func;
};

// Iterative QPE workflow to estimate the energy of a Hamiltonian operator.
// For the first pass, we implement this as a workflow.
// This can be integrated as a CostFuncEvaluator if needed.
class IterativeQpeWorkflow : public QuantumSimulationWorkflow {
public:
  // Translate/stretch the Hamiltonian operator for QPE.
  struct HamOpConverter {
    double translation;
    double stretch;
    void fromObservable(Observable *obs);
    std::shared_ptr<Observable> stretchObservable(Observable *obs) const;
    double computeEnergy(double phaseVal) const;
  };

  virtual bool initialize(const HeterogeneousMap &params) override;
  virtual QuantumSimulationResult
  execute(const QuantumSimulationModel &model) override;
  virtual const std::string name() const override { return "iqpe"; }
  virtual const std::string description() const override { return ""; }

  static std::shared_ptr<CompositeInstruction>
  constructQpeTrotterCircuit(std::shared_ptr<Observable> obs,
                             double trotter_step, size_t nbQubits,
                             int steps = 1, int k = 1, double omega = 0);

private:
  std::shared_ptr<CompositeInstruction>
  constructQpeCircuit(std::shared_ptr<Observable> obs, int k, double omega,
                      bool measure = true) const;

private:
  // Number of time slices (>=1)
  int num_steps;
  // Number of iterations (>=1)
  int num_iters;
  HamOpConverter ham_converter;
};

class DefaultObjFuncEval : public CostFunctionEvaluator {
public:
  // Evaluate the cost
  virtual double
  evaluate(std::shared_ptr<CompositeInstruction> state_prep) override;
  virtual const std::string name() const override { return "default"; }
  virtual const std::string description() const override { return ""; }
};

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