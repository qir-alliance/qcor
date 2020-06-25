#pragma once

#include "AcceleratorBuffer.hpp"
#include "qcor.hpp"

namespace qcor {
static constexpr double pi = 3.141592653589793238;
namespace __internal__ {
// This simple struct is a way for us to
// enumerate commonly seen TranslationFunctors, a utility
// that maps quantum kernel argument structure to the
// qcor Optimizer / OptFunction std::vector<double> x parameters.
struct TranslationFunctorGenerator {

  qcor::TranslationFunctor<qreg, double> operator()(qreg &q,
                                                    std::tuple<double> &&);
  qcor::TranslationFunctor<qreg, std::vector<double>>
  operator()(qreg &q, std::tuple<std::vector<double>> &&);
};

class CountRotationAngles {
public:
  std::size_t &count;
  CountRotationAngles(std::size_t &c) : count(c) {}
  void operator()(std::vector<double> tuple_element_vec) {
    count += tuple_element_vec.size();
  }
  void operator()(double tuple_element_double) { count++; }
};

} // namespace __internal__

// High-level VQE class, enables programmers to
// easily construct the VQE task given an parameterized
// qcor quantum kernel and the Hamiltonian / Observable of
// interest
template <typename QuantumKernel> class VQE {
protected:
  // Reference to the paramerized
  // quantum kernel functor
  QuantumKernel &ansatz;

  // Reference to the Hamiltonian / Observable,
  // will dictate measurements on the kernel
  Observable &observable;

  // We need the user to tell us the number of
  // parameters in the parameterized circuit
  std::size_t n_params = 0;

  // Register of qubits to operate on
  qreg q;

  GradientEvaluator grad_eval;

public:
  // Typedef for describing the energy / params return type
  using VQEResultType = std::pair<double, std::vector<double>>;

  // Typedef describing all seen energies and corresponding parameters
  using VQEEnergiesAndParameters =
      std::vector<std::pair<double, std::vector<double>>>;

  // Constructor
  VQE(QuantumKernel &kernel, Observable &obs)
      : ansatz(kernel), observable(obs) {
    q = qalloc(obs.nBits());
  }

  // Constructor, with gradient evaluator specification
  VQE(QuantumKernel &kernel, Observable &obs, GradientEvaluator &geval)
      : ansatz(kernel), observable(obs), grad_eval(geval) {
    q = qalloc(obs.nBits());
  }
  
  // Execute the VQE task synchronously, assumes default optimizer
  template <typename... Args> VQEResultType execute(Args... initial_params) {
    auto optimizer = qcor::createOptimizer("nlopt");
    auto handle = execute_async<Args...>(optimizer, initial_params...);
    return this->sync(handle);
  }

  // Execute the VQE task asynchronously, default optimizer
  template <typename... Args> Handle execute_async(Args... initial_params) {
    auto optimizer = qcor::createOptimizer("nlopt");
    return execute_async<Args...>(optimizer, initial_params...);
  }

  // Execute the VQE task synchronously, use provided Optimizer
  template <typename... Args>
  VQEResultType execute(std::shared_ptr<Optimizer> optimizer,
                        Args... initial_params) {
    auto handle = execute_async<Args...>(optimizer, initial_params...);
    return this->sync(handle);
  }

  // Execute the VQE task asynchronously, use provided Optimizer
  template <typename... Args>
  Handle execute_async(std::shared_ptr<Optimizer> optimizer,
                       Args... initial_params) {
    // Get the VQE ObjectiveFunction and set the qreg
    auto objective = qcor::createObjectiveFunction("vqe", ansatz, observable);
    objective->set_qreg(q);

    // Convert input args to a tuple
    auto init_args_tuple = std::make_tuple(initial_params...);

    // Set the Optimizer initial parameters
    std::vector<double> init_params;
    __internal__::ConvertDoubleLikeToVectorDouble build_up_init_params(
        init_params);
    __internal__::tuple_for_each(init_args_tuple, build_up_init_params);
    optimizer->appendOption("initial-parameters", init_params);

    // Create the Arg Translator
    __internal__::TranslationFunctorGenerator gen;
    auto arg_translator = gen(q, std::tuple<Args...>());

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

// Next, add QAOA...

} // namespace qcor
