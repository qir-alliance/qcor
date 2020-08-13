#pragma once

#include "AcceleratorBuffer.hpp"
#include "objective_function.hpp"
#include "qcor.hpp"
#include "qcor_utils.hpp"
#include "qrt.hpp"
#include <memory>
#include <xacc_internal_compiler.hpp>

namespace qcor {
static constexpr double pi = 3.141592653589793238;
namespace __internal__ {
// This simple struct is a way for us to
// enumerate commonly seen TranslationFunctors, a utility
// that maps quantum kernel argument structure to the
// qcor Optimizer / OptFunction std::vector<double> x parameters.
struct TranslationFunctorAutoGenerator {

  qcor::TranslationFunctor<qreg, double> operator()(qreg &q,
                                                    std::tuple<double> &&);
  qcor::TranslationFunctor<qreg, std::vector<double>>
  operator()(qreg &q, std::tuple<std::vector<double>> &&);

  template <typename... Args>
  qcor::TranslationFunctor<qreg, Args...> operator()(qreg &q,
                                                     std::tuple<Args...> &&) {
    return
        [](const std::vector<double>) { return std::tuple<qreg, Args...>(); };
  }
};

// Utility class to count the number of rotation
// angles in a parameterized circuit evaluation
class CountRotationAngles {
public:
  std::size_t &count;
  CountRotationAngles(std::size_t &c) : count(c) {}
  void operator()(std::vector<double> tuple_element_vec) {
    count += tuple_element_vec.size();
  }
  void operator()(double tuple_element_double) { count++; }
  template <typename T> void operator()(T &tuple_element) {}
};

} // namespace __internal__

// High-level VQE class, enables programmers to
// easily construct the VQE task given an parameterized
// qcor quantum kernel and the Hamiltonian / Observable of
// interest
template <typename... KernelArgs> class VQE {

protected:
  inline static const std::string OBJECTIVE_NAME = "vqe";
  inline static const std::string OPTIMIZER_INIT_PARAMS = "initial-parameters";
  inline static const std::string DEFAULT_OPTIMIZER = "nlopt";

  // Reference to the paramerized
  // quantum kernel functor as a void pointer
  void *ansatz_ptr;

  // Reference to the Hamiltonian / Observable,
  // will dictate measurements on the kernel
  Observable &observable;

  // We need the user to tell us the number of
  // parameters in the parameterized circuit
  std::size_t n_params = 0;

  // Register of qubits to operate on
  qreg q;

  // Any holding user-specified qcor::TranslationFunctor<qreg, Args...>
  // Using any here to keep us from having to
  // template VQE class any further.
  std::any translation_functor;

  HeterogeneousMap options;

public:
  // Typedef for describing the energy / params return type
  using VQEResultType = std::pair<double, std::vector<double>>;

  // Typedef describing all seen energies and corresponding parameters
  using VQEEnergiesAndParameters =
      std::vector<std::pair<double, std::vector<double>>>;

  // Constructor
  VQE(void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                     qreg, KernelArgs...),
      Observable &obs, HeterogeneousMap opts = {})
      : ansatz_ptr(reinterpret_cast<void *>(quantum_kernel_functor)),
        observable(obs) {
    q = qalloc(obs.nBits());
    options = opts;
  }

  // Constructor, takes a TranslationFunctor as a general
  // template type that we store to the protected any member
  // and cast later
  template <typename TranslationFunctorT>
  VQE(void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                     qreg, KernelArgs...),
      Observable &obs, TranslationFunctorT &&tfunc, HeterogeneousMap opts = {})
      : ansatz_ptr(reinterpret_cast<void *>(quantum_kernel_functor)),
        observable(obs), translation_functor(tfunc) {
    q = qalloc(obs.nBits());
    options = opts;
  }

  // Execute the VQE task synchronously, assumes default optimizer
  VQEResultType execute(KernelArgs... initial_params) {
    auto optimizer = qcor::createOptimizer(DEFAULT_OPTIMIZER);
    auto handle = execute_async(optimizer, initial_params...);
    return this->sync(handle);
  }

  // Execute the VQE task asynchronously, default optimizer
  Handle execute_async(KernelArgs... initial_params) {
    auto optimizer = qcor::createOptimizer(DEFAULT_OPTIMIZER);
    return execute_async(optimizer, initial_params...);
  }

  // Execute the VQE task synchronously, use provided Optimizer
  //   template <typename... Args>
  VQEResultType execute(std::shared_ptr<Optimizer> optimizer,
                        KernelArgs... initial_params) {
    auto handle = execute_async(optimizer, initial_params...);
    return this->sync(handle);
  }

  // Execute the VQE task asynchronously, use provided Optimizer
  //   template <typename... Args>
  Handle execute_async(std::shared_ptr<Optimizer> optimizer,
                       KernelArgs... initial_params) {

    auto ansatz_casted =
        reinterpret_cast<void (*)(std::shared_ptr<CompositeInstruction>, qreg,
                                  KernelArgs...)>(ansatz_ptr);

    // objective->set_qreg(q);

    // Convert input args to a tuple
    auto init_args_tuple = std::make_tuple(initial_params...);

    // Set the Optimizer initial parameters
    std::vector<double> init_params;
    __internal__::ConvertDoubleLikeToVectorDouble build_up_init_params(
        init_params);
    __internal__::tuple_for_each(init_args_tuple, build_up_init_params);
    optimizer->appendOption(OPTIMIZER_INIT_PARAMS, init_params);

    // Create the Arg Translator
    TranslationFunctor<qreg, KernelArgs...> arg_translator;
    if (translation_functor.has_value()) {
      arg_translator = std::any_cast<TranslationFunctor<qreg, KernelArgs...>>(
          translation_functor);
    } else {
      __internal__::TranslationFunctorAutoGenerator auto_gen;
      arg_translator = auto_gen(q, std::tuple<KernelArgs...>());
    }

    // Count all rotation angles (n parameters)
    __internal__::CountRotationAngles count_params(n_params);
    __internal__::tuple_for_each(init_args_tuple, count_params);

    // Get the VQE ObjectiveFunction and set the qreg
    auto objective = qcor::createObjectiveFunction(
        ansatz_casted, observable,
        std::make_shared<ArgsTranslator<qreg, KernelArgs...>>(arg_translator),
        q, n_params);

    if (optimizer->isGradientBased() &&
        !options.stringExists("gradient-strategy")) {
      options.insert("gradient-strategy", "central");
    }
    options.insert("observable", __internal__::qcor_as_shared(&observable));
    objective->set_options(options);

    // Run TaskInitiate, kick of the VQE job asynchronously
    return qcor::taskInitiate(objective, optimizer);
  }

  // Sync up the results with the host thread
  VQEResultType sync(Handle &h) {
    auto results = qcor::sync(h);
    return std::make_pair(results.opt_val, results.opt_params);
  }

  // Return all unique parameter sets this VQE run used
  std::vector<std::vector<double>> get_unique_parameters() {
    auto tmp_ei = q.results()->getAllUnique("parameters");
    std::vector<std::vector<double>> ret;
    for (auto &ei : tmp_ei) {
      ret.push_back(ei.template as<std::vector<double>>());
    }
    return ret;
  }

  // Return all energies seen at their corresponding parameter sets
  VQEEnergiesAndParameters get_unique_energies() {
    auto tmp_ei = q.results()->getAllUnique("qcor-params-energy");
    VQEEnergiesAndParameters ret;
    for (auto &ei : tmp_ei) {
      auto raw = ei.template as<std::vector<double>>();
      std::vector<double> params(raw.begin(), raw.begin() + raw.size() - 1);
      double energy = *(raw.end() - 1);
      ret.push_back(std::make_pair(energy, params));
    }
    return ret;
  }
};

namespace __internal__ {
void qaoa_ansatz(qreg q, int n_steps, std::vector<double> gamma,
                 std::vector<double> beta, std::string cost_ham_str) {
  void __internal_call_function_qaoa_ansatz(qreg, int, std::vector<double>,
                                            std::vector<double>, std::string);
  __internal_call_function_qaoa_ansatz(q, n_steps, gamma, beta, cost_ham_str);
}
class qaoa_ansatz
    : public qcor::QuantumKernel<class qaoa_ansatz, qreg, int,
                                 std::vector<double>, std::vector<double>,
                                 std::string> {
  friend class qcor::QuantumKernel<class qaoa_ansatz, qreg, int,
                                   std::vector<double>, std::vector<double>,
                                   std::string>;

protected:
  void operator()(qreg q, int n_steps, std::vector<double> gamma,
                  std::vector<double> beta, std::string cost_ham_str) {
    // quantum::set_backend("qpp");
    if (!parent_kernel) {
      parent_kernel = qcor::__internal__::create_composite(kernel_name);
      q.setNameAndStore("qreg_dCRJkswbUA");
    }
    quantum::set_current_program(parent_kernel);
    auto nQubits = q.size();
    int gamma_counter = 0;
    int beta_counter = 0;
    for (int i = 0; i < nQubits; i++) {
      quantum::h(q[0]);
    }
    auto cost_ham_ptr = createObservable(cost_ham_str);
    auto &cost_ham = *cost_ham_ptr.get();
    auto cost_terms = cost_ham.getNonIdentitySubTerms();
    for (int step = 0; step < n_steps; step++) {
      for (int i = 0; i < cost_terms.size(); i++) {
        auto cost_term = cost_terms[i];
        auto m_gamma = gamma[gamma_counter];
        quantum::exp(q, m_gamma, cost_term);
        gamma_counter++;
      }
      for (int i = 0; i < nQubits; i++) {
        auto ref_ham_term = X(i);
        auto m_beta = beta[beta_counter];
        quantum::exp(q, m_beta, ref_ham_term);
        beta_counter++;
      }
    }
  }

public:
  inline static const std::string kernel_name = "qaoa_ansatz";
  qaoa_ansatz(qreg q, int n_steps, std::vector<double> gamma,
              std::vector<double> beta, std::string cost_ham_str)
      : QuantumKernel<qaoa_ansatz, qreg, int, std::vector<double>,
                      std::vector<double>, std::string>(q, n_steps, gamma, beta,
                                                        cost_ham_str) {}
  qaoa_ansatz(std::shared_ptr<qcor::CompositeInstruction> _parent, qreg q,
              int n_steps, std::vector<double> gamma, std::vector<double> beta,
              std::string cost_ham_str)
      : QuantumKernel<qaoa_ansatz, qreg, int, std::vector<double>,
                      std::vector<double>, std::string>(
            _parent, q, n_steps, gamma, beta, cost_ham_str) {}
  virtual ~qaoa_ansatz() {
    if (disable_destructor) {
      return;
    }
    auto [q, n_steps, gamma, beta, cost_ham_str] = args_tuple;
    operator()(q, n_steps, gamma, beta, cost_ham_str);
    if (optimize_only) {
      xacc::internal_compiler::execute_pass_manager();
      return;
    }
    if (is_callable) {
      quantum::submit(q.results());
    }
  }
};
void qaoa_ansatz(std::shared_ptr<qcor::CompositeInstruction> parent, qreg q,
                 int n_steps, std::vector<double> gamma,
                 std::vector<double> beta, std::string cost_ham_str) {
  class qaoa_ansatz k(parent, q, n_steps, gamma, beta, cost_ham_str);
}
void __internal_call_function_qaoa_ansatz(qreg q, int n_steps,
                                          std::vector<double> gamma,
                                          std::vector<double> beta,
                                          std::string cost_ham_str) {
  class qaoa_ansatz k(q, n_steps, gamma, beta, cost_ham_str);
}
} // namespace __internal__

// QAOA provides a high-level data structure
// for driving the qcor API in a way that affects the
// typical QAOA algorithm. Users instantiate this class
// by providing the cost hamiltonian and number of qaoa steps,
// and optionally the reference hamiltonian. Running the
// algorithm is then just simply invoking the execute() method
class QAOA {
protected:
  // The number of qaoa steps
  std::size_t nSteps = 1;

  // The cost hamiltonian
  PauliOperator &cost;

  // The reference hamiltonian
  PauliOperator ref;

  // The qubit register to run on
  qreg q;

  HeterogeneousMap options;

  // utility for delegating to xacc error call
  // implemented in qcor_hybrid.cpp to avoid
  // xacc.hpp include here
  void error(const std::string &message);

public:
  // Helper qaoa algorithm result type
  using QAOAResultType = std::pair<double, std::vector<double>>;

  // The Constructionr, takes the cost hamiltonian and the number of steps.
  // will assume reference hamiltonian of an X pauli on all qubits
  QAOA(PauliOperator &obs, const std::size_t _n_steps,
       HeterogeneousMap opts = {})
      : cost(obs), nSteps(_n_steps), ref(PauliOperator()) {
    for (int i = 0; i < cost.nBits(); i++) {
      ref += qcor::X(i);
    }
    options = opts;
  }

  // The constructor, takes cost and reference hamiltonian and number of steps
  QAOA(PauliOperator &obs, PauliOperator &ref_ham, const std::size_t _n_steps,
       HeterogeneousMap opts = {})
      : cost(obs), nSteps(_n_steps), ref(ref_ham) {
    options = opts;
  }

  // Return the number of gamma parameters
  auto n_gamma() { return cost.getNonIdentitySubTerms().size(); }
  // Return the number of beta parameters
  auto n_beta() { return ref.getNonIdentitySubTerms().size(); }
  // Return the number of qaoa steps
  auto n_steps() { return nSteps; }
  // Return the total number of parameters
  auto n_parameters() { return n_steps() * (n_gamma() + n_beta()); }

  // Execute the algorithm synchronously, optionally provide the initial
  // parameters. Will fail if initial_parameters.size() != n_parameters()
  QAOAResultType execute(const std::vector<double> initial_parameters = {}) {
    auto optimizer = qcor::createOptimizer("nlopt");
    return execute(optimizer, initial_parameters);
  }

  // Execute the algorithm synchronously, provding an Optimizer and optionally
  // initial parameters Will fail if initial_parameters.size() != n_parameters()
  QAOAResultType execute(std::shared_ptr<Optimizer> optimizer,
                         const std::vector<double> initial_parameters = {}) {
    auto handle = execute_async(optimizer, initial_parameters);
    auto results = this->sync(handle);
    return std::make_pair(results.first, results.second);
  }

  // Execute the algorithm asynchronously, provding an Optimizer and optionally
  // initial parameters Will fail if initial_parameters.size() != n_parameters()
  Handle execute_async(std::shared_ptr<Optimizer> optimizer,
                       const std::vector<double> initial_parameters = {}) {

    q = qalloc(cost.nBits());
    auto n_params = nSteps * (n_gamma() + n_beta());

    std::vector<double> init_params;
    if (initial_parameters.empty()) {
      init_params = qcor::random_vector(0.0, 1.0, n_params);
    } else {
      if (initial_parameters.size() != n_parameters()) {
        error("invalid initial_parameters provided, " +
              std::to_string(n_parameters()) +
              " != " + std::to_string(initial_parameters.size()));
      }
      init_params = initial_parameters;
    }

    auto args_translator =
        std::make_shared<ArgsTranslator<qreg, int, std::vector<double>,
                                        std::vector<double>, std::string>>(
            [&](const std::vector<double> x) {
              // split x into gamma and beta sets
              int nGamma = cost.getNonIdentitySubTerms().size();
              int nBeta = ref.getNonIdentitySubTerms().size();
              std::vector<double> gamma(x.begin(), x.begin() + nSteps * nGamma),
                  beta(x.begin() + nSteps * nGamma,
                       x.begin() + nSteps * nGamma + nSteps * nBeta);
              return std::make_tuple(q, nSteps, gamma, beta, cost.toString());
            });

    auto objective = qcor::createObjectiveFunction(
        __internal__::qaoa_ansatz, cost, args_translator, q, n_params);

    if (optimizer->isGradientBased() &&
        !options.stringExists("gradient-strategy")) {
      options.insert("gradient-strategy", "central");
    }

    Observable &obs = cost;
    options.insert("observable", __internal__::qcor_as_shared(&obs));
    objective->set_options(options);
    optimizer->appendOption("initial-parameters", init_params);

    return qcor::taskInitiate(objective, optimizer);
  }

  // Sync up the results with the host thread
  QAOAResultType sync(Handle &h) {
    auto results = qcor::sync(h);
    return std::make_pair(results.opt_val, results.opt_params);
  }
}; // namespace qcor

void execute_adapt(qreg q, const HeterogeneousMap &&m);

template <typename... KernelArgs> class ADAPT {
protected:
  Observable &observable;
  void *state_prep_ptr;
  std::shared_ptr<Optimizer> optimizer;
  HeterogeneousMap options;

  // Register of qubits to operate on
  qreg q;

public:
  ADAPT(void (*state_prep_kernel)(std::shared_ptr<CompositeInstruction>, qreg,
                                  KernelArgs...),
        Observable &obs, std::shared_ptr<Optimizer> opt,
        HeterogeneousMap _options)
      : state_prep_ptr(reinterpret_cast<void *>(state_prep_kernel)),
        observable(obs), optimizer(opt), options(_options) {
    q = qalloc(obs.nBits());
  }

  double execute(KernelArgs... initial_args) {
    auto state_prep_casted =
        reinterpret_cast<void (*)(std::shared_ptr<CompositeInstruction>, qreg,
                                  KernelArgs...)>(state_prep_ptr);
    auto parent_composite =
        qcor::__internal__::create_composite("adapt_composite");
    state_prep_casted(parent_composite, q, initial_args...);
    // parent_composite now has the circuit in it

    auto accelerator = xacc::internal_compiler::get_qpu();

    options.insert("initial-state", parent_composite);
    options.insert("observable", &observable);
    options.insert("optimizer", optimizer);
    options.insert("accelerator", accelerator);

    execute_adapt(q, std::move(options));

    return q.results()->getInformation("opt-val").template as<double>();
  }
};

void execute_qite(qreg q, const HeterogeneousMap &&m);
// Next, QITE
template <typename... KernelArgs> class QITE {
protected:
  Observable &observable;
  void *state_prep_ptr;
  const int n_steps;
  const double step_size;
  // Register of qubits to operate on
  qreg q;

public:
  QITE(void (*state_prep_kernel)(std::shared_ptr<CompositeInstruction>, qreg,
                                 KernelArgs...),
       Observable &obs, const int _n_steps, const double _step_size)
      : state_prep_ptr(reinterpret_cast<void *>(state_prep_kernel)),
        observable(obs), n_steps(_n_steps), step_size(_step_size) {
    q = qalloc(obs.nBits());
  }

  double execute(KernelArgs... initial_args) {
    auto state_prep_casted =
        reinterpret_cast<void (*)(std::shared_ptr<CompositeInstruction>, qreg,
                                  KernelArgs...)>(state_prep_ptr);
    auto parent_composite =
        qcor::__internal__::create_composite("qite_composite");
    state_prep_casted(parent_composite, q, initial_args...);
    // parent_composite now has the circuit in it

    auto accelerator = xacc::internal_compiler::get_qpu();
    execute_qite(q, {{"steps", n_steps},
                     {"step-size", step_size},
                     {"ansatz", parent_composite},
                     {"accelerator", accelerator},
                     {"observable", &observable}});
    return q.results()->getInformation("opt-val").template as<double>();
  }
};
} // namespace qcor
