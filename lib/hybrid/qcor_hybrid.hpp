#pragma once

#include "AcceleratorBuffer.hpp"
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

  // The GradientEvaluator to use if
  // we are given an Optimizer that is gradient-based
  GradientEvaluator grad_eval;

  // Any holding user-specified qcor::TranslationFunctor<qreg, Args...>
  // Using any here to keep us from having to
  // template VQE class any further.
  std::any translation_functor;

public:
  // Typedef for describing the energy / params return type
  using VQEResultType = std::pair<double, std::vector<double>>;

  // Typedef describing all seen energies and corresponding parameters
  using VQEEnergiesAndParameters =
      std::vector<std::pair<double, std::vector<double>>>;

  // Constructor
  VQE(void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                     qreg, KernelArgs...),
      Observable &obs)
      : ansatz_ptr(reinterpret_cast<void *>(quantum_kernel_functor)),
        observable(obs) {
    q = qalloc(obs.nBits());
  }

  // Constructor, with gradient evaluator specification
  VQE(void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                     qreg, KernelArgs...),
      Observable &obs, GradientEvaluator &geval)
      : ansatz_ptr(reinterpret_cast<void *>(quantum_kernel_functor)),
        observable(obs), grad_eval(geval) {
    q = qalloc(obs.nBits());
  }

  // Constructor, takes a TranslationFunctor as a general
  // template type that we store to the protected any member
  // and cast later
  template <typename TranslationFunctorT>
  VQE(void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                     qreg, KernelArgs...),
      Observable &obs, TranslationFunctorT &&tfunc)
      : ansatz_ptr(reinterpret_cast<void *>(quantum_kernel_functor)),
        observable(obs), translation_functor(tfunc) {
    q = qalloc(obs.nBits());
  }

  // Constructor, takes a TranslationFunctor as a general
  // template type that we store to the protected any member
  // and cast later. Also takes gradient evaluator
  template <typename TranslationFunctorT>
  VQE(void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                     qreg, KernelArgs...),
      Observable &obs, GradientEvaluator &geval, TranslationFunctorT &&tfunc)
      : ansatz_ptr(reinterpret_cast<void *>(quantum_kernel_functor)),
        observable(obs), translation_functor(tfunc), grad_eval(geval) {
    q = qalloc(obs.nBits());
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

    // Get the VQE ObjectiveFunction and set the qreg
    auto objective = qcor::createObjectiveFunction(OBJECTIVE_NAME,
                                                   ansatz_casted, observable);
    objective->set_qreg(q);

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
      std::cout << "Using this arg translator\n";
      arg_translator = std::any_cast<TranslationFunctor<qreg, KernelArgs...>>(
          translation_functor);
    } else {
      __internal__::TranslationFunctorAutoGenerator auto_gen;
      arg_translator = auto_gen(q, std::tuple<KernelArgs...>());
    }

    // Count all rotation angles (n parameters)
    __internal__::CountRotationAngles count_params(n_params);
    __internal__::tuple_for_each(init_args_tuple, count_params);

    if (optimizer->isGradientBased()) {

      if (!grad_eval) {
        grad_eval = [&, arg_translator, objective](const std::vector<double> x,
                                                   std::vector<double> &dx) {
          for (int i = 0; i < dx.size(); i++) {
            auto xplus = x[i] + pi / 2.;
            auto xminus = x[i] - pi / 2.;
            std::vector<double> tmpx_plus = x, tmpx_minus = x;
            tmpx_plus[i] = xplus;
            tmpx_minus[i] = xminus;

            auto translated_tuple_xp = arg_translator(tmpx_plus);
            auto translated_tuple_xm = arg_translator(tmpx_minus);

            auto results1 =
                qcor::__internal__::call(objective, translated_tuple_xp);
            auto results2 =
                qcor::__internal__::call(objective, translated_tuple_xm);

            dx[i] = 0.5 * (results1 - results2);
          }
        };
      }

      return qcor::taskInitiate(objective, optimizer, grad_eval, arg_translator,
                                n_params);

    } else {
      // Run TaskInitiate, kick of the VQE job asynchronously
      return qcor::taskInitiate(objective, optimizer, arg_translator, n_params);
    }
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

  // The GradientEvaluator to use if
  // we are given an Optimizer that is gradient-based
  GradientEvaluator grad_eval;

  // The cost hamiltonian
  PauliOperator &cost;

  // The reference hamiltonian
  PauliOperator ref;

  // The qubit register to run on
  qreg q;

  // Scale factor for parameter shift gradient rule
  double ps_scale_factor = 1.0;

  // Internal helper function for creating
  // the quantum kernel from an xasm string
  void initial_compile_qaoa_code();

  // Reference to the qaoa parameterized circuit
  std::shared_ptr<CompositeInstruction> qaoa_circuit;

  // XASM QAOA code for creating the qaoa circuit
  inline static const std::string qaoa_xasm_code =
      R"#(__qpu__ void qaoa_ansatz(qreg q, int n, std::vector<double> betas, std::vector<double> gammas, qcor::PauliOperator& costHamiltonian, qcor::PauliOperator& refHamiltonian) {
  qaoa(q, n, betas, gammas, costHamiltonian, refHamiltonian);
})#";

  // utility for delegating to xacc error call
  // implemented in qcor_hybrid.cpp to avoid
  // xacc.hpp include here
  void error(const std::string &message);

public:
  // Helper qaoa algorithm result type
  using QAOAResultType = std::pair<double, std::vector<double>>;

  // The Constructionr, takes the cost hamiltonian and the number of steps.
  // will assume reference hamiltonian of an X pauli on all qubits
  QAOA(PauliOperator &obs, const std::size_t _n_steps)
      : cost(obs), nSteps(_n_steps), ref(PauliOperator()) {
    for (int i = 0; i < cost.nBits(); i++) {
      ref += qcor::X(i);
    }
    initial_compile_qaoa_code();
  }

  // The constructor, takes cost and reference hamiltonian and number of steps
  QAOA(PauliOperator &obs, PauliOperator &ref_ham, const std::size_t _n_steps)
      : cost(obs), nSteps(_n_steps), ref(ref_ham) {
    initial_compile_qaoa_code();
  }

  // The Constructionr, takes the cost hamiltonian and the number of steps.
  // will assume reference hamiltonian of an X pauli on all qubits
  // also provide gradient evaluator
  QAOA(PauliOperator &obs, GradientEvaluator &geval, const std::size_t _n_steps)
      : cost(obs), nSteps(_n_steps), ref(PauliOperator()), grad_eval(geval) {
    for (int i = 0; i < cost.nBits(); i++) {
      ref += qcor::X(i);
    }
    initial_compile_qaoa_code();
  }

  // The constructor, takes cost and reference hamiltonian and number of steps
  // Also provide gradient evaluator
  QAOA(PauliOperator &obs, PauliOperator &ref_ham, GradientEvaluator &geval,
       const std::size_t _n_steps)
      : cost(obs), nSteps(_n_steps), ref(ref_ham), grad_eval(geval) {
    initial_compile_qaoa_code();
  }

  // Return the number of gamma parameters
  auto n_gamma() { return cost.getNonIdentitySubTerms().size(); }
  // Return the number of beta parameters
  auto n_beta() { return ref.getNonIdentitySubTerms().size(); }
  // Return the number of qaoa steps
  auto n_steps() { return nSteps; }
  // Return the total number of parameters
  auto n_parameters() { return n_steps() * (n_gamma() + n_beta()); }

  // Provide the scale factor for the parameter shift rule
  void set_parameter_shift_scale_factor(const double scale) {
    if (scale < 0.0) {
      error("invalid parameter shift scale factor, must be greater than 0.0");
    }
    ps_scale_factor = scale;
  }
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

    auto objective = qcor::createObjectiveFunction("vqe", qaoa_circuit, cost);
    objective->set_qreg(q);
    optimizer->appendOption("initial-parameters", init_params);

    // See if we are using a gradient-based optimizer
    const auto use_gradients = optimizer->isGradientBased();

    return qcor::taskInitiate(
        objective, optimizer,
        [objective, use_gradients, this](const std::vector<double> x,
                                         std::vector<double> &dx) {
          // Set the size of the beta vector
          const auto n_betas = nSteps * q.size();

          // Utility to split x into beta and gamma vecs
          auto create_betas_gammas = [n_betas](const std::vector<double> x) {
            std::vector<double> betas;
            std::vector<double> gammas;
            for (int i = 0; i < n_betas; ++i) {
              betas.emplace_back(x[i]);
            }
            for (int i = betas.size(); i < x.size(); ++i) {
              gammas.emplace_back(x[i]);
            }

            return std::make_pair(betas, gammas);
          };

          // If we need gradients, evaluate them
          // with our default parameter shift or the
          // provide GradientEvaluator
          if (use_gradients) {
            if (!grad_eval) {
              // Parameter shift rule...
              for (int i = 0; i < dx.size(); i++) {
                auto xplus = x[i] + ps_scale_factor * pi / 2.;
                auto xminus = x[i] - ps_scale_factor * pi / 2.;
                std::vector<double> tmpx_plus = x, tmpx_minus = x;
                tmpx_plus[i] = xplus;
                tmpx_minus[i] = xminus;

                // Don't print the verbose output for gradients...
                const auto cached_verbose = qcor::get_verbose();
                qcor::set_verbose(false);

                // evaluate at x[i] + scale * pi / 2
                auto [betas_p, gammas_p] = create_betas_gammas(tmpx_plus);
                auto results1 =
                    (*objective)(q, q.size(), betas_p, gammas_p, cost, ref);

                // evaluate at x[i] - scale * pi / 2
                auto [betas_m, gammas_m] = create_betas_gammas(tmpx_minus);
                auto results2 =
                    (*objective)(q, q.size(), betas_m, gammas_m, cost, ref);
                qcor::set_verbose(cached_verbose);

                // set the gradient element
                dx[i] = 0.5 * (results1 - results2);
              }
            } else {
              // evaluate with the custom evaluator
              grad_eval(x, dx);
            }
          }

          // Evaluate at x and return
          auto [betas, gammas] = create_betas_gammas(x);
          return (*objective)(q, q.size(), betas, gammas, cost, ref);
        },
        n_params);
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
  std::shared_ptr<Observable> observable;
  void *state_prep_ptr;
  const std::string pool, subAlgo;
  const int nElectrons;
  std::shared_ptr<Optimizer> optimizer;
  // Register of qubits to operate on
  qreg q;

public:
  ADAPT(void (*state_prep_kernel)(std::shared_ptr<CompositeInstruction>, qreg,
                                  KernelArgs...),
        std::shared_ptr<Observable> obs, const int _ne, const std::string _pool,
        const std::string _subAlgo, std::shared_ptr<Optimizer> opt)
      : state_prep_ptr(reinterpret_cast<void *>(state_prep_kernel)),
        observable(obs), nElectrons(_ne), pool(_pool), subAlgo(_subAlgo),
        optimizer(opt) {
    q = qalloc(obs->nBits());
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
    execute_adapt(q, {{"n-electrons", nElectrons},
                      {"pool", pool},
                      {"initial-state", parent_composite},
                      {"optimizer", optimizer},
                      {"sub-algorithm", subAlgo},
                      {"accelerator", accelerator},
                      {"observable", observable}});
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
