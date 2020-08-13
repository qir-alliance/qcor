#pragma once

#include "qcor_observable.hpp"
#include "qcor_utils.hpp"
#include "quantum_kernel.hpp"
#include <functional>

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

  void update_kernel(std::shared_ptr<CompositeInstruction> updated_kernel) {
    kernel = updated_kernel;
  }

  void update_current_iterate_parameters(std::vector<double> x) {
    current_iterate_parameters = x;
  }
  // Set any extra options needed for the objective function
  virtual void set_options(HeterogeneousMap &opts) { options = opts; }
  template <typename T> void update_options(const std::string key, T value) {
    options.insert(key, value);
  }

  // this really shouldnt be called.
  virtual xacc::internal_compiler::qreg get_qreg() {
    throw std::bad_function_call();
    return qalloc(1);
  }
};

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
  xacc::internal_compiler::qreg &qreg;

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
  void set_options(HeterogeneousMap &opts) override {
    options = opts;
    helper->set_options(opts);
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
        kernel_evaluator = [&](std::vector<double> x)
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
    return (*helper)(qreg, dx);
  }

  // Return the qreg
  xacc::internal_compiler::qreg get_qreg() override { return qreg; }

  // Provide a name and description
  const std::string name() const override { return "objective-impl"; }
  const std::string description() const override { return ""; }
};

namespace __internal__ {
// Get the objective function from the service registry
std::shared_ptr<ObjectiveFunction> get_objective(const std::string &type);

template <typename T> std::shared_ptr<T> qcor_as_shared(T *t) {
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

  std::shared_ptr<ArgsTranslator<qreg, std::vector<double>>>
  operator()(qreg &q, std::tuple<qreg, std::vector<double>> &&) {
    return std::make_shared<ArgsTranslator<qreg, std::vector<double>>>(
        [&](const std::vector<double> &x) { return std::make_tuple(q, x); });
  }

  template <typename... DoubleTypes>
  std::shared_ptr<ArgsTranslator<qreg, DoubleTypes...>>
  operator()(qreg &q, std::tuple<qreg, DoubleTypes...> &&t) {
    if constexpr ((std::is_same<DoubleTypes, double>::value && ...)) {
      return std::make_shared<ArgsTranslator<qreg, DoubleTypes...>>(
          [&](const std::vector<double> &x) {
            auto qreg_tuple = std::make_tuple(q);
            auto double_tuple = create_tuple<sizeof...(DoubleTypes)>(x);
            return std::tuple_cat(qreg_tuple, double_tuple);
          });
    } else {
      error("QCOR cannot auto-generate a ArgsTranslator for this "
            "ObjectiveFunction. Please provide a custom ArgsTranslator to "
            "createObjectiveFunction.");
      return std::make_shared<ArgsTranslator<qreg, DoubleTypes...>>(
          [&](const std::vector<double> &x) { return t; });
    }
  }
};

} // namespace __internal__

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
  for (int i = 0; i < q.size(); i++)
    all_zs.insert({i, "Z"});
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
  for (int i = 0; i < q.size(); i++)
    all_zs.insert({i, "Z"});
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
  for (int i = 0; i < q.size(); i++)
    all_zs.insert({i, "Z"});
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
  for (int i = 0; i < q.size(); i++)
    all_zs.insert({i, "Z"});
  auto observable = std::make_shared<PauliOperator>(all_zs);
  return std::make_shared<ObjectiveFunctionImpl<Args...>>(
      kernel_ptr, observable, q, args_translator, helper, nParams, options);
}

} // namespace qcor