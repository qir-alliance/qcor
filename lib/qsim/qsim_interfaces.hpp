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
struct QsimModel {
  // Model name.
  std::string name;
  
  // The Observable operator that needs to be measured/minimized. 
  Observable *observable;
  
  // The system Hamiltonian which can be static or dynamic (time-dependent).
  // This can be the same or different from the observable operator.
  TdObservable hamiltonian;
};

// Generic model builder (factory)
// Create a model which capture the problem description.
class ModelBuilder {
public:
  // ======== Direct model builder ==============
  // Strongly-typed parameters/argument.
  // Build a simple Hamiltonian-based model: static Hamiltonian which is also
  // the observable of interest.
  static QsimModel createModel(Observable *obs, const HeterogeneousMap &params = {});
  // Build a time-dependent problem model:
  //  -  obs: observable operator to measure.
  //  -  td_ham: time-dependent Hamiltonian to evolve the system.
  //     e.g. a function to map from time to Hamiltonian operator.
  static QsimModel createModel(Observable* obs, TdObservable td_ham,const HeterogeneousMap &params = {});

  // ========  High-level model builder ==============
  // The idea here is to have contributed modules to translate problem descriptions
  // in various formats, e.g. PyScf, Psi4, broombridge, etc. 
  // into QCOR's Observable type.
  // Inputs:
  //  - format: key to look up module/plugin to digest the input data.
  //  - data: model descriptions in plain-text format, e.g. load from file.
  static QsimModel createModel(const std::string &format,
                               const std::string &data,
                               const HeterogeneousMap &params = {});
};

// Quantum Simulation Workflow (Protocol)
// This can handle both variational workflow (optimization loop)
// as well as simple Trotter evolution workflow.
// Result is stored in a HetMap
using QsimResult = HeterogeneousMap;
// Abstract workflow:
class QsimWorkflow : Identifiable {
public:
  virtual bool initialize(const HeterogeneousMap &params) = 0;
  virtual QsimResult execute(const QsimModel &model) = 0;

protected:
  CostFunctionEvaluator *evaluator;
};

// Supported Workflow: to be defined and implemented
enum class WorkFlow { VQE, TD, PE };

// TODO: create a service registry for these.
QsimWorkflow* getWorkflow(WorkFlow type, const HeterogeneousMap &init_params); 

} // namespace qcor