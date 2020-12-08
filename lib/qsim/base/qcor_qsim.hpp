#pragma once
#include "Accelerator.hpp"
#include "AcceleratorBuffer.hpp"
#include "Circuit.hpp"
#include "kernel_evaluator.hpp"
#include "objective_function.hpp"
#include "qcor_observable.hpp"
#include "qcor_optimizer.hpp"
#include "qcor_utils.hpp"
#include "qrt.hpp"
#include "quantum_kernel.hpp"
#include "taskInitiate.hpp"
#include <memory>
#include <qalloc>
#include <xacc_internal_compiler.hpp>

using CompositeInstruction = xacc::CompositeInstruction;
using Accelerator = xacc::Accelerator;
using Identifiable = xacc::Identifiable;

namespace qcor {
namespace qsim {
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
class CostFunctionEvaluator : public Identifiable {
public:
  // Evaluate the cost
  virtual double evaluate(std::shared_ptr<CompositeInstruction> state_prep) = 0;
  // Batching evaluation: observing multiple kernels in batches.
  // E.g. for non-vqe cases (Trotter), we have all kernels ready for observable evaluation
  virtual std::vector<double> evaluate(
      std::vector<std::shared_ptr<CompositeInstruction>> state_prep_circuits) {
    // Default is one-by-one, subclass to provide batching if supported.
    std::vector<double> result;
    for (auto &circuit : state_prep_circuits) {
      result.emplace_back(evaluate(circuit));
    }
    return result;
  }
  
  virtual bool initialize(Observable *observable,
                          const HeterogeneousMap &params = {});

protected:
  Observable *target_operator;
  HeterogeneousMap hyperParams;
};

// Trotter time-dependent simulation workflow
// Time-dependent Hamiltonian is a function mapping from time t to Observable
// operator.
using TdObservable = std::function<PauliOperator(double)>;

// Capture a quantum chemistry problem.
// TODO: generalize this to capture all potential use cases.

struct QuantumSimulationModel {
  QuantumSimulationModel() : observable(nullptr) {}
  // Model name.
  std::string name;

  // The Observable operator that needs to be measured/minimized.
  Observable *observable;

  // The system Hamiltonian which can be static or dynamic (time-dependent).
  // This can be the same or different from the observable operator.
  TdObservable hamiltonian;

  // QuantumSimulationModel also support a user-defined (fixed) ansatz.
  std::shared_ptr<KernelFunctor> user_defined_ansatz;
};

// Generic model builder (factory)
// Create a model which capture the problem description.
class ModelBuilder {
public:
  // Generic Heisenberg model
  struct HeisenbergModel {
    double Jx = 0.0;
    double Jy = 0.0;
    double Jz = 0.0;
    double h_ext = 0.0;
    // Support for H_BAR normalization
    double H_BAR = 1.0;
    // "X", "Y", or "Z"
    std::string ext_dir = "Z";
    int num_spins = 2;
    std::vector<int> initial_spins;
    // Time-dependent freq.
    // Default to using the cosine function.
    double freq = 0.0;
    // User-provided custom time-dependent function:
    std::function<double(double)> time_func;
    // Allows a simple Pythonic kwarg-style initialization.
    // i.e. all params have preset defaults, only update those that are
    // specified.
    void fromDict(const HeterogeneousMap &params);

    bool validateModel() const {
      const bool ext_dir_valid = (ext_dir == "X" || ext_dir == "Y" || ext_dir == "Z");
      const bool initial_spins_valid = (initial_spins.empty() || (initial_spins.size() == num_spins));
      return ext_dir_valid && initial_spins_valid;
    }
  };

  // ======== Direct model builder ==============
  // Strongly-typed parameters/argument.
  // Build a simple Hamiltonian-based model: static Hamiltonian which is also
  // the observable of interest.
  static QuantumSimulationModel createModel(Observable *obs,
                                           const HeterogeneousMap &params = {});
  static QuantumSimulationModel
  createModel(PauliOperator &obs, const HeterogeneousMap &params = {}) {
    return createModel(&obs, params);
  }
  
  // Build a time-dependent problem model:
  //  -  obs: observable operator to measure.
  //  -  td_ham: time-dependent Hamiltonian to evolve the system.
  //     e.g. a function to map from time to Hamiltonian operator.
  static QuantumSimulationModel createModel(Observable *obs, TdObservable td_ham,
                                           const HeterogeneousMap &params = {});
  // Pauli operator overload:
  static QuantumSimulationModel
  createModel(PauliOperator &obs, TdObservable td_ham,
              const HeterogeneousMap &params = {}) {
    return createModel(&obs, td_ham, params);
  }

  // ========  High-level model builder ==============
  // The idea here is to have contributed modules to translate problem
  // descriptions in various formats, e.g. PyScf, Psi4, broombridge, etc. into
  // QCOR's Observable type. Inputs:
  //  - format: key to look up module/plugin to digest the input data.
  //  - data: model descriptions in plain-text format, e.g. load from file.
  //  - params: extra parameters to pass to the parser/generator,
  //            e.g. any transformations required in order to generate the
  //            Observable.
  static QuantumSimulationModel createModel(const std::string &format,
                                           const std::string &data,
                                           const HeterogeneousMap &params = {});
  // Predefined model type that we support intrinsically.
  enum class ModelType { Heisenberg };
  static QuantumSimulationModel createModel(ModelType type,
                                           const HeterogeneousMap &params);
  // ========== QuantumSimulationModel with a fixed (pre-defined) ansatz ========
  // The ansatz is provided as a QCOR kernel.
  template <typename... Args>
  static inline QuantumSimulationModel createModel(
      void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                     Args...),
      Observable *obs, size_t nbQubits, size_t nbParams) {
    auto kernel_functor =
        createKernelFunctor(quantum_kernel_functor, nbQubits, nbParams);

    QuantumSimulationModel model;
    model.observable = obs;
    model.user_defined_ansatz = kernel_functor;
    return model;
  }

  template <typename... Args>
  static inline QuantumSimulationModel createModel(
      void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                     Args...),
      PauliOperator &obs, size_t nbQubits, size_t nbParams) {
    return createModel(quantum_kernel_functor, &obs, nbQubits, nbParams);
  }

  // Passing the state-preparation ansatz as a CompositeInstruction
  static inline QuantumSimulationModel
  createModel(std::shared_ptr<CompositeInstruction> composite,
              Observable *obs) {
    QuantumSimulationModel model;
    model.observable = obs;
    model.user_defined_ansatz = createKernelFunctor(composite);
    return model;
  }

  static inline QuantumSimulationModel
  createModel(std::shared_ptr<CompositeInstruction> composite,
              PauliOperator &obs) {
    return createModel(composite, &obs);
  }
};

// Quantum Simulation Workflow (Protocol)
// This can handle both variational workflow (optimization loop)
// as well as simple Trotter evolution workflow.
// Result is stored in a HetMap
using QuantumSimulationResult = HeterogeneousMap;
// Abstract workflow:
class QuantumSimulationWorkflow : public Identifiable {
public:
  virtual bool initialize(const HeterogeneousMap &params) = 0;
  virtual QuantumSimulationResult
  execute(const QuantumSimulationModel &model) = 0;

protected:
  std::shared_ptr<CostFunctionEvaluator> evaluator;
};

// Get workflow by name:
std::shared_ptr<QuantumSimulationWorkflow>
getWorkflow(const std::string &name, const HeterogeneousMap &init_params);

// Get the Obj (cost) function evaluator:
std::shared_ptr<CostFunctionEvaluator>
getObjEvaluator(Observable *observable, const std::string &name = "default",
                const HeterogeneousMap &init_params = {});
inline std::shared_ptr<CostFunctionEvaluator>
getObjEvaluator(PauliOperator &obs, const std::string &name = "default",
                const HeterogeneousMap &init_params = {}) {
  return getObjEvaluator(&obs, name, init_params);
}

// Helper to apply optimization/placement before evaluation:
void executePassManager(
    std::vector<std::shared_ptr<CompositeInstruction>> evalKernels);
} // namespace qsim
} // namespace qcor