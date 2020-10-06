#pragma once
#include "Accelerator.hpp"
#include "AcceleratorBuffer.hpp"
#include "Circuit.hpp"
#include "qcor.hpp"
#include "qcor_utils.hpp"
#include "qrt.hpp"
#include <memory>
#include <xacc_internal_compiler.hpp>
using namespace xacc;

namespace qcor {

// Struct captures a state-preparation circuit.
struct Ansatz {
  std::shared_ptr<CompositeInstruction> circuit;
  std::vector<std::string> symbol_names;
  size_t nb_qubits;
};

// State-preparation method (i.e. ansatz):
// For example,
//  - Real time Hamiltonian evolution with Trotter approximation (of various
//  orders)
//  - Real time Hamiltonian evolution with Taylor series/LCU approach
//  - Imaginary time evolution with QITE
//  - Thermo-field doubles state preparation
//  - Two-point correlation function measurements
//  - QFT
class AnsatzGenerator {
public:
  virtual Ansatz create_ansatz(Observable *obs = nullptr,
                               const HeterogeneousMap &params = {}) = 0;
};

// CostFunctionEvaluator take an unknown quantum state (the circuit that
// prepares the unknown state) and a target operator (e.g. Hamiltonian
// Observable) as input and produce a estimation as output.
class CostFunctionEvaluator {
public:
  // Evaluate the cost
  virtual double evaluate(std::shared_ptr<CompositeInstruction> state_prep);
  virtual bool initialize(const HeterogeneousMap &params);
  static CostFunctionEvaluator *getInstance();

protected:
  Observable *target_operator;
  Accelerator *quantum_backend;
  HeterogeneousMap hyperParams;

private:
  static inline CostFunctionEvaluator *instance = nullptr;
};

// Quantum Simulation Workflow (Protocol)
// This can handle both variational workflow (optimization loop)
// as well as simple Trotter evolution workflow.
// Result is stored in a HetMap
using QsimResult = HeterogeneousMap;
// Abstract workflow:
class QsimWorkflow {
public:
  virtual QsimResult execute() = 0;
  virtual bool initialize(const HeterogeneousMap &params) = 0;

protected:
  CostFunctionEvaluator *evaluator;
};

// Supported Workflow: to be defined and implemented
enum class WorkFlow { VQE, TD, PE };
// Quantum Simulation Model: to capture the problem description and the workflow.
// TODO: support high-level problem descriptions.
class QsimModel {
public:
  bool initialize(WorkFlow workflow_type,
                          const HeterogeneousMap &params);
  QsimWorkflow *get_workflow() { return qsim_workflow; }

private:
  WorkFlow type;
  QsimWorkflow *qsim_workflow;
};

// Generic model builder (factory)
// Create a model which capture the problem statement as well as the simulation
// workflow.
class ModelBuilder {
public:
  static QsimModel createModel(const HeterogeneousMap &params);
};


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
  virtual QsimResult execute() override;
  virtual bool initialize(const HeterogeneousMap &params) override;

private:
  Optimizer *optimizer;
};

// Trotter time-dependent simulation workflow
// Time-dependent Hamiltonian is a function mapping from time t to Observable
// operator.
using TdObservable = std::function<PauliOperator(double)>;

// Time-dependent evolution workflow which can handle
// time-dependent Hamiltonian operator.
class TimeDependentWorkflow : public QsimWorkflow {
public:
  virtual QsimResult execute() override;
  virtual bool initialize(const HeterogeneousMap &params) override;
  static QsimWorkflow *getInstance();

private:
  static inline TimeDependentWorkflow *instance = nullptr;
  double t_0;
  double t_final;
  double dt;
  TdObservable ham_func;
};
} // namespace qcor