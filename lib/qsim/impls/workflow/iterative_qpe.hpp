#pragma once
#include "qcor_qsim.hpp"

namespace qcor {
namespace qsim {
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

  static std::shared_ptr<CompositeInstruction> constructQpeTrotterCircuit(
      std::shared_ptr<Observable> obs, double trotter_step, size_t nbQubits,
      double compensatedAncRot = 0, int steps = 1, int k = 1, double omega = 0);

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
} // namespace qsim
} // namespace qcor