#pragma once

#include <functional>

#include "gradient_function.hpp"
#include "qcor_observable.hpp"
#include "qcor_utils.hpp"
#include "quantum_kernel.hpp"

namespace qcor {

class ObjectiveFunction : public xacc::OptFunction, public xacc::Identifiable {
 protected:
  // Quantum kernel function pointer, we will use
  // this to cast to kernel(composite, args...).
  // Doing it this way means we don't template ObjectiveFunction
  void *kernel_ptr;

  // CompositeInstruction representation of the
  // evaluated quantum kernel
  std::shared_ptr<CompositeInstruction> kernel;

  // non-owning
  std::shared_ptr<Observable> observable;

  HeterogeneousMap options;
  std::vector<double> current_iterate_parameters;

 public:
  double operator()(const std::vector<double> &x) {
    std::vector<double> unused_grad;
    return operator()(x, unused_grad);
  }
  double operator()(const std::vector<double> &x,
                    std::vector<double> &dx) override {
    throw std::bad_function_call();
    return 0.0;
  }
  virtual double operator()(xacc::internal_compiler::qreg &qreg,
                            std::vector<double> &dx) = 0;

  void update_observable(std::shared_ptr<Observable> updated_observable) {
    observable = updated_observable;
  }

  std::shared_ptr<Observable> get_observable() { return observable; }

  void update_kernel(std::shared_ptr<CompositeInstruction> updated_kernel) {
    kernel = updated_kernel;
  }

  void update_current_iterate_parameters(std::vector<double> x) {
    current_iterate_parameters = x;
  }
  // Set any extra options needed for the objective function
  virtual void set_options(HeterogeneousMap &opts) { options = opts; }
  template <typename T>
  void update_options(const std::string key, T value) {
    options.insert(key, value);
  }

  // this really shouldnt be called.
  virtual xacc::internal_compiler::qreg get_qreg() {
    throw std::bad_function_call();
    return qalloc(1);
  }

  virtual std::function<
      std::shared_ptr<CompositeInstruction>(std::vector<double>)>
  get_kernel_evaluator() {
    // Derive (ObjectiveFunctionImpl) only
    error("Illegal call to get_kernel_evaluator().");
    return {};
  };
};

namespace __internal__ {
// Get the objective function from the service registry
std::shared_ptr<ObjectiveFunction> get_objective(const std::string &type);

template <typename T>
std::shared_ptr<T> qcor_as_shared(T *t) {
  return std::shared_ptr<T>(t, [](T *const) {});
}

template <std::size_t... Is>
auto create_tuple_impl(std::index_sequence<Is...>,
                       const std::vector<double> &arguments) {
  return std::make_tuple(arguments[Is]...);
}

template <std::size_t N>
auto create_tuple(const std::vector<double> &arguments) {
  return create_tuple_impl(std::make_index_sequence<N>{}, arguments);
}

struct ArgsTranslatorAutoGenerator {
  std::shared_ptr<ArgsTranslator<qreg, std::vector<double>>> operator()(
      qreg &q, std::tuple<qreg, std::vector<double>> &&) {
    return std::make_shared<ArgsTranslator<qreg, std::vector<double>>>(
        [&](const std::vector<double> &x) { return std::make_tuple(q, x); });
  }

  template <typename... DoubleTypes>
  std::shared_ptr<ArgsTranslator<qreg, DoubleTypes...>> operator()(
      qreg &q, std::tuple<qreg, DoubleTypes...> &&t) {
    if constexpr ((std::is_same<DoubleTypes, double>::value && ...)) {
      return std::make_shared<ArgsTranslator<qreg, DoubleTypes...>>(
          [&](const std::vector<double> &x) {
            auto qreg_tuple = std::make_tuple(q);
            auto double_tuple = create_tuple<sizeof...(DoubleTypes)>(x);
            return std::tuple_cat(qreg_tuple, double_tuple);
          });
    } else {
      error(
          "QCOR cannot auto-generate a ArgsTranslator for this "
          "ObjectiveFunction. Please provide a custom ArgsTranslator to "
          "createObjectiveFunction.");
      return std::make_shared<ArgsTranslator<qreg, DoubleTypes...>>(
          [&](const std::vector<double> &x) { return t; });
    }
  }
};

}  // namespace __internal__

template <typename... KernelArgs>
class ObjectiveFunctionImpl : public ObjectiveFunction {
 private:
  using LocalArgsTranslator = ArgsTranslator<KernelArgs...>;

  std::shared_ptr<CompositeInstruction> create_new_composite() {
    // Create a composite that we can pass to the functor
    std::stringstream name_ss;
    name_ss << this << "_qkernel";
    auto _kernel = qcor::__internal__::create_composite(name_ss.str());
    return _kernel;
  }

 protected:
  std::shared_ptr<LocalArgsTranslator> args_translator;
  std::shared_ptr<ObjectiveFunction> helper;
  xacc::internal_compiler::qreg qreg;
  std::shared_ptr<GradientFunction> gradiend_method;
  // Kernel evaluator from qpu_lambda
  std::optional<
      std::function<std::shared_ptr<CompositeInstruction>(std::vector<double>)>>
      lambda_kernel_evaluator;

public:
  ObjectiveFunctionImpl(void *k_ptr, std::shared_ptr<Observable> obs,
                        xacc::internal_compiler::qreg &qq,
                        std::shared_ptr<LocalArgsTranslator> translator,
                        std::shared_ptr<ObjectiveFunction> obj_helper,
                        const int dim, HeterogeneousMap opts)
      : qreg(qq) {
    kernel_ptr = k_ptr;
    observable = obs;
    args_translator = translator;
    helper = obj_helper;
    _dim = dim;
    _function = *this;
    options = opts;
    options.insert("observable", observable);
    helper->update_observable(observable);
    helper->set_options(options);
  }

  ObjectiveFunctionImpl(void *k_ptr, std::shared_ptr<Observable> obs,
                        std::shared_ptr<ObjectiveFunction> obj_helper,
                        const int dim, HeterogeneousMap opts) {
    qreg = ::qalloc(obs->nBits());
    kernel_ptr = k_ptr;
    observable = obs;
    __internal__::ArgsTranslatorAutoGenerator auto_gen;
    args_translator = auto_gen(qreg, std::tuple<KernelArgs...>());
    // args_translator = translator;
    helper = obj_helper;
    _dim = dim;
    _function = *this;
    options = opts;
    options.insert("observable", observable);
    helper->update_observable(observable);
    helper->set_options(options);
  }

  ObjectiveFunctionImpl(
      std::function<void(std::shared_ptr<CompositeInstruction>, KernelArgs...)>
          &functor,
      std::shared_ptr<Observable> obs, xacc::internal_compiler::qreg &qq,
      std::shared_ptr<LocalArgsTranslator> translator,
      std::shared_ptr<ObjectiveFunction> obj_helper, const int dim,
      HeterogeneousMap opts)
      : qreg(qq) {
    // std::cout << "Constructed from lambda\n";
    lambda_kernel_evaluator =
        [&, functor](
            std::vector<double> x) -> std::shared_ptr<CompositeInstruction> {
      // std::cout << "HOWDY:\n";
      // Create a new CompositeInstruction, and create a tuple
      // from it so we can concatenate with the tuple args
      auto m_kernel = create_new_composite();
      auto kernel_composite_tuple = std::make_tuple(m_kernel);

      // Translate x parameters into kernel args (represented as a tuple)
      auto translated_tuple = (*args_translator)(x);

      // Concatenate the two to make the args list (kernel, args...)
      auto concatenated =
          std::tuple_cat(kernel_composite_tuple, translated_tuple);
      std::apply(functor, concatenated);
      // std::cout << m_kernel->toString() << "\n";
      return m_kernel;
    };
    observable = obs;
    args_translator = translator;
    helper = obj_helper;
    _dim = dim;
    _function = *this;
    options = opts;
    options.insert("observable", observable);
    helper->update_observable(observable);
    helper->set_options(options);
  }

  void set_options(HeterogeneousMap &opts) override {
    options = opts;
    helper->set_options(opts);
  }

  // Construct the kernel evaluator of this ObjectiveFunctionImpl
  std::function<std::shared_ptr<CompositeInstruction>(std::vector<double>)> get_kernel_evaluator() override {
    // Turn kernel evaluation into a functor that we can use here
    // and share with the helper ObjectiveFunction for gradient evaluation
    std::function<std::shared_ptr<CompositeInstruction>(std::vector<double>)>
        kernel_evaluator = lambda_kernel_evaluator.has_value()
                               ? lambda_kernel_evaluator.value()
                               : [&](std::vector<double> x)
        -> std::shared_ptr<CompositeInstruction> {
      // Define a function pointer type for the quantum kernel
      void (*kernel_functor)(std::shared_ptr<CompositeInstruction>,
                             KernelArgs...);
      // Cast to the function pointer type
      if (kernel_ptr) {
        kernel_functor = reinterpret_cast<void (*)(
            std::shared_ptr<CompositeInstruction>, KernelArgs...)>(kernel_ptr);
      }
      // Create a new CompositeInstruction, and create a tuple
      // from it so we can concatenate with the tuple args
      auto m_kernel = create_new_composite();
      auto kernel_composite_tuple = std::make_tuple(m_kernel);

      // Translate x parameters into kernel args (represented as a tuple)
      auto translated_tuple = (*args_translator)(x);

      // Concatenate the two to make the args list (kernel, args...)
      auto concatenated =
          std::tuple_cat(kernel_composite_tuple, translated_tuple);

      // Call the functor with those arguments
      qcor::__internal__::evaluate_function_with_tuple_args(kernel_functor,
                                                            concatenated);
      return m_kernel;
    };

    return kernel_evaluator;
  }

  // This will not be called on this class... It will only be called
  // on helpers...
  double operator()(xacc::internal_compiler::qreg &qreg,
                    std::vector<double> &dx) override {
    throw std::bad_function_call();
    return 0.0;
  }

  double operator()(const std::vector<double> &x,
                    std::vector<double> &dx) override {
    current_iterate_parameters = x;
    helper->update_current_iterate_parameters(x);

    // Define a function pointer type for the quantum kernel
    void (*kernel_functor)(std::shared_ptr<CompositeInstruction>,
                           KernelArgs...);
    // Cast to the function pointer type
    if (kernel_ptr) {
      kernel_functor = reinterpret_cast<void (*)(
          std::shared_ptr<CompositeInstruction>, KernelArgs...)>(kernel_ptr);
    }

    // Turn kernel evaluation into a functor that we can use here
    // and share with the helper ObjectiveFunction for gradient evaluation
    std::function<std::shared_ptr<CompositeInstruction>(std::vector<double>)>
        kernel_evaluator = lambda_kernel_evaluator.has_value()
                               ? lambda_kernel_evaluator.value()
                               : [&](std::vector<double> x)
        -> std::shared_ptr<CompositeInstruction> {
      // Create a new CompositeInstruction, and create a tuple
      // from it so we can concatenate with the tuple args
      auto m_kernel = create_new_composite();
      auto kernel_composite_tuple = std::make_tuple(m_kernel);

      // Translate x parameters into kernel args (represented as a tuple)
      auto translated_tuple = (*args_translator)(x);

      // Concatenate the two to make the args list (kernel, args...)
      auto concatenated =
          std::tuple_cat(kernel_composite_tuple, translated_tuple);

      // Call the functor with those arguments
      qcor::__internal__::evaluate_function_with_tuple_args(kernel_functor,
                                                            concatenated);
      return m_kernel;
    };

    // Give the kernel evaluator to the helper
    helper->update_options("kernel-evaluator", kernel_evaluator);

    // Kernel is set / evaluated... run sub-type operator()
    kernel = kernel_evaluator(x);
    helper->update_kernel(kernel);

    // Save the input dx:
    const auto input_dx = dx;
    
    auto cost_val = (*helper)(qreg, dx);
    // If we needs gradients:
    // the optimizer requires dx (not empty)
    // and the concrete ObjFunc sub-class doesn't calculate the gradients.
    if (!dx.empty() && input_dx == dx) {
      if (dx.size() != x.size()) {
        error("Dimension mismatched: gradients and parameters vectors have "
              "different size.");
      }

      if (!gradiend_method) {
        std::string gradient_method_name =
            qcor::__internal__::DEFAULT_GRADIENT_METHOD;
        // Backward compatible:
        // If the "gradient-strategy" was specified in the option.
        if (options.stringExists("gradient-strategy")) {
          gradient_method_name = options.getString("gradient-strategy");
        }
        gradiend_method = qcor::__internal__::get_gradient_method(
            gradient_method_name, xacc::as_shared_ptr(this), options);
      }

      dx = (*gradiend_method)(x, cost_val);
    }
    return cost_val;
  }

  // Return the qreg
  xacc::internal_compiler::qreg get_qreg() override { return qreg; }

  // Provide a name and description
  const std::string name() const override { return "objective-impl"; }
  const std::string description() const override { return ""; }
};

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    std::shared_ptr<Observable> observable, qreg &q, const int nParams,
    HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective("vqe");
  __internal__::ArgsTranslatorAutoGenerator auto_gen;
  auto args_translator = auto_gen(q, std::tuple<Args...>());

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, observable, q, args_translator, helper, nParams, options);
}

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string obj_name,
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    std::shared_ptr<Observable> observable, qreg &q, const int nParams,
    HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective(obj_name);
  __internal__::ArgsTranslatorAutoGenerator auto_gen;
  auto args_translator = auto_gen(q, std::tuple<Args...>());

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, observable, q, args_translator, helper, nParams, options);
}

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    Observable &observable, qreg &q, const int nParams,
    HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective("vqe");
  __internal__::ArgsTranslatorAutoGenerator auto_gen;
  auto args_translator = auto_gen(q, std::tuple<Args...>());

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, __internal__::qcor_as_shared(&observable), q, args_translator,
      helper, nParams, options);
}

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string obj_name,
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    Observable &observable, qreg &q, const int nParams,
    HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective(obj_name);
  __internal__::ArgsTranslatorAutoGenerator auto_gen;
  auto args_translator = auto_gen(q, std::tuple<Args...>());

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, __internal__::qcor_as_shared(&observable), q, args_translator,
      helper, nParams, options);
}

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    qreg &q, const int nParams, HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective("vqe");
  __internal__::ArgsTranslatorAutoGenerator auto_gen;
  auto args_translator = auto_gen(q, std::tuple<Args...>());

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  std::map<int, std::string> all_zs;
  for (int i = 0; i < q.size(); i++) all_zs.insert({i, "Z"});
  auto observable = std::make_shared<PauliOperator>(all_zs);
  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, observable, q, args_translator, helper, nParams, options);
}

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string obj_name,
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    qreg &q, const int nParams, HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective(obj_name);
  __internal__::ArgsTranslatorAutoGenerator auto_gen;
  auto args_translator = auto_gen(q, std::tuple<Args...>());

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  std::map<int, std::string> all_zs;
  for (int i = 0; i < q.size(); i++) all_zs.insert({i, "Z"});
  auto observable = std::make_shared<PauliOperator>(all_zs);
  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, observable, q, args_translator, helper, nParams, options);
}

/////// Now provide overloads with ArgsTranslators

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    std::shared_ptr<Observable> observable,
    std::shared_ptr<ArgsTranslator<Args...>> args_translator, qreg &q,
    const int nParams, HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective("vqe");

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, observable, q, args_translator, helper, nParams, options);
}

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string obj_name,
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    std::shared_ptr<Observable> observable,
    std::shared_ptr<ArgsTranslator<Args...>> args_translator, qreg &q,
    const int nParams, HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective(obj_name);

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, observable, q, args_translator, helper, nParams, options);
}

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    Observable &observable,
    std::shared_ptr<ArgsTranslator<Args...>> args_translator, qreg &q,
    const int nParams, HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective("vqe");

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, __internal__::qcor_as_shared(&observable), q, args_translator,
      helper, nParams, options);
}

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string obj_name,
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    Observable &observable,
    std::shared_ptr<ArgsTranslator<Args...>> args_translator, qreg &q,
    const int nParams, HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective(obj_name);

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, __internal__::qcor_as_shared(&observable), q, args_translator,
      helper, nParams, options);
}

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    std::shared_ptr<ArgsTranslator<Args...>> args_translator, qreg &q,
    const int nParams, HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective("vqe");

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  std::map<int, std::string> all_zs;
  for (int i = 0; i < q.size(); i++) all_zs.insert({i, "Z"});
  auto observable = std::make_shared<PauliOperator>(all_zs);
  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, observable, q, args_translator, helper, nParams, options);
}

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string obj_name,
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    std::shared_ptr<ArgsTranslator<Args...>> args_translator, qreg &q,
    const int nParams, HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective(obj_name);

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  std::map<int, std::string> all_zs;
  for (int i = 0; i < q.size(); i++) all_zs.insert({i, "Z"});
  auto observable = std::make_shared<PauliOperator>(all_zs);
  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, observable, q, args_translator, helper, nParams, options);
}

/// no qreg args

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    std::shared_ptr<Observable> observable, const int nParams,
    HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective("vqe");

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, observable, helper, nParams, options);
}

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string obj_name,
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    std::shared_ptr<Observable> observable, const int nParams,
    HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective(obj_name);

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, observable, helper, nParams, options);
}

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    Observable &observable, const int nParams,
    HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective("vqe");

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, __internal__::qcor_as_shared(&observable), helper, nParams,
      options);
}

template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string obj_name,
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    Observable &observable, const int nParams,
    HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective(obj_name);

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, __internal__::qcor_as_shared(&observable), helper, nParams,
      options);
}

template <typename... CaptureArgs, typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    _qpu_lambda<CaptureArgs...> &lambda,
    std::shared_ptr<ArgsTranslator<Args...>> args_translator,
    Observable &observable, qreg &q, const int nParams,
    HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective("vqe");
  std::function<void(std::shared_ptr<CompositeInstruction>, Args...)> kernel_fn =
      [&lambda](std::shared_ptr<CompositeInstruction> comp, Args... args) -> void {
        return lambda.eval_with_parent(comp, args...);
      };

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_fn, __internal__::qcor_as_shared(&observable), q,
      args_translator, helper, nParams, options);
}

// Create ObjFunc from a qpu_lambda w/o a specific args_translater
// Assume the lambda has a VQE-compatible signature
template <typename... CaptureArgs>
std::shared_ptr<ObjectiveFunction>
createObjectiveFunction(_qpu_lambda<CaptureArgs...> &lambda,
                        Observable &observable, qreg &q, const int nParams,
                        HeterogeneousMap &&options = {}) {
  if (lambda.var_type ==
      _qpu_lambda<CaptureArgs...>::Variational_Arg_Type::None) {
    error("qpu_lambda has an incompatible signature. Please provide an "
          "ArgsTranslator.");
  }
  auto helper = qcor::__internal__::get_objective("vqe");
  std::function<void(std::shared_ptr<CompositeInstruction>, qreg,
                     std::vector<double>)>
      kernel_fn = [&lambda](std::shared_ptr<CompositeInstruction> comp, qreg q,
                            std::vector<double> params) -> void {
    if (lambda.var_type ==
        _qpu_lambda<CaptureArgs...>::Variational_Arg_Type::Vec_Double) {
      return lambda.eval_with_parent(comp, q, params);
    }
    if (lambda.var_type ==
        _qpu_lambda<CaptureArgs...>::Variational_Arg_Type::Double) {
      if (params.size() != 1) {
        error("Invalid number of parameters. Expected 1, got " +
              std::to_string(params.size()));
      }
      return lambda.eval_with_parent(comp, q, params[0]);
    }
    error("Internal error: invalid qpu lambda type encountered.");
  };

  auto args_translator =
      std::make_shared<ArgsTranslator<qreg, std::vector<double>>>(
          [&](const std::vector<double> x) { return std::make_tuple(q, x); });

  return std::make_shared<ObjectiveFunctionImpl<qreg, std::vector<double>>>(
      kernel_fn, __internal__::qcor_as_shared(&observable), q, args_translator,
      helper, nParams, options);
}

// Objective function with gradient options:
// Generic method: user provides a gradient calculation method.
template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    std::shared_ptr<Observable> observable, qreg &q, const int nParams,
    std::shared_ptr<GradientFunction> gradient_method,
    HeterogeneousMap &&options = {}) {
  auto helper = qcor::__internal__::get_objective("vqe");
  __internal__::ArgsTranslatorAutoGenerator auto_gen;
  auto args_translator = auto_gen(q, std::tuple<Args...>());

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, observable, q, args_translator, helper, nParams, options);
}
} // namespace qcor