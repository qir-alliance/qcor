#pragma once
#include "Accelerator.hpp"
#include "AcceleratorBuffer.hpp"
#include "Circuit.hpp"
#include "qcor.hpp"
#include "qcor_utils.hpp"
#include "qrt.hpp"
#include <memory>
#include <xacc_internal_compiler.hpp>
using CompositeInstruction = xacc::CompositeInstruction;
using Accelerator = xacc::Accelerator;
using Identifiable = xacc::Identifiable;

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
class AnsatzGenerator : public Identifiable {
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
  virtual bool initialize(Observable *observable,
                          const HeterogeneousMap &params = {});
  static CostFunctionEvaluator *getInstance();

protected:
  Observable *target_operator;
  Accelerator *quantum_backend;
  HeterogeneousMap hyperParams;

private:
  static inline CostFunctionEvaluator *instance = nullptr;
};

// Trotter time-dependent simulation workflow
// Time-dependent Hamiltonian is a function mapping from time t to Observable
// operator.
using TdObservable = std::function<PauliOperator(double)>;

// Capture a quantum chemistry problem.
// TODO: generalize this to capture all potential use cases.
struct QuatumSimulationModel {
  // Model name.
  std::string name;

  // The Observable operator that needs to be measured/minimized.
  Observable *observable;

  // The system Hamiltonian which can be static or dynamic (time-dependent).
  // This can be the same or different from the observable operator.
  TdObservable hamiltonian;
  
  // QuatumSimulationModel also support a user-defined (fixed) ansatz.
  std::shared_ptr<CompositeInstruction> user_defined_ansatz;
};

// Generic model builder (factory)
// Create a model which capture the problem description.
class ModelBuilder {
public:
  // ======== Direct model builder ==============
  // Strongly-typed parameters/argument.
  // Build a simple Hamiltonian-based model: static Hamiltonian which is also
  // the observable of interest.
  static QuatumSimulationModel createModel(Observable *obs,
                               const HeterogeneousMap &params = {});
  // Build a time-dependent problem model:
  //  -  obs: observable operator to measure.
  //  -  td_ham: time-dependent Hamiltonian to evolve the system.
  //     e.g. a function to map from time to Hamiltonian operator.
  static QuatumSimulationModel createModel(Observable *obs, TdObservable td_ham,
                               const HeterogeneousMap &params = {});

  // ========  High-level model builder ==============
  // The idea here is to have contributed modules to translate problem
  // descriptions in various formats, e.g. PyScf, Psi4, broombridge, etc. into
  // QCOR's Observable type. Inputs:
  //  - format: key to look up module/plugin to digest the input data.
  //  - data: model descriptions in plain-text format, e.g. load from file.
  //  - params: extra parameters to pass to the parser/generator,
  //            e.g. any transformations required in order to generate the Observable.
  static QuatumSimulationModel createModel(const std::string &format,
                               const std::string &data,
                               const HeterogeneousMap &params = {});

  // ========== QuatumSimulationModel with a fixed (pre-defined) ansatz ========
  // The ansatz is provided as a QCOR kernel.
  template <typename... Args>
  static inline QuatumSimulationModel createModel(
      void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                     Args...),
      Observable *obs) {
    QuatumSimulationModel model;
    std::cout << "HOWDY\n";
    return model;
  }
};

// Quantum Simulation Workflow (Protocol)
// This can handle both variational workflow (optimization loop)
// as well as simple Trotter evolution workflow.
// Result is stored in a HetMap
using QuatumSimulationResult = HeterogeneousMap;
// Abstract workflow:
class QuatumSimulationWorkflow : public Identifiable {
public:
  virtual bool initialize(const HeterogeneousMap &params) = 0;
  virtual QuatumSimulationResult execute(const QuatumSimulationModel &model) = 0;

protected:
  CostFunctionEvaluator *evaluator;
};

// Get workflow by name:
std::shared_ptr<QuatumSimulationWorkflow> getWorkflow(const std::string &name,
                                          const HeterogeneousMap &init_params);
} // namespace qcor