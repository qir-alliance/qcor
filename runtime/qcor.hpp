#ifndef RUNTIME_QCOR_HPP_
#define RUNTIME_QCOR_HPP_

#include <functional>
#include <future>
#include <memory>
#include <tuple>

#include "CompositeInstruction.hpp"
#include "Observable.hpp"
#include "Optimizer.hpp"

#include "qalloc"
#include "xacc_internal_compiler.hpp"
#include "PauliOperator.hpp"

#include "qrt.hpp"

namespace qcor {

using OptFunction = xacc::OptFunction;
using HeterogeneousMap = xacc::HeterogeneousMap;
using Observable = xacc::Observable;
using Optimizer = xacc::Optimizer;
using CompositeInstruction = xacc::CompositeInstruction;
using PauliOperator = xacc::quantum::PauliOperator;


PauliOperator X(int idx){
  return PauliOperator({{idx, "X"}});
}

PauliOperator Y(int idx){
  return PauliOperator({{idx, "Y"}});
}

PauliOperator Z(int idx){
  return PauliOperator({{idx, "Z"}});
}
template<typename T>
PauliOperator operator+(T coeff, PauliOperator &op){
  return PauliOperator(coeff) + op;
}
template<typename T>
PauliOperator operator+(PauliOperator &op, T coeff){
  return PauliOperator(coeff) + op;
}

template<typename T>
PauliOperator operator-(T coeff, PauliOperator &op){
  return -1.0*coeff + op;
}

template<typename T>
PauliOperator operator-(PauliOperator &op, T coeff){
  return -1.0*coeff + op;
}

PauliOperator sig_plus(int idx){
  std::complex<double> imag (0.0, 1.0);
  return X(idx) + imag * Y(idx);
}

PauliOperator sig_minus(int idx){
  std::complex<double> imag (0.0, -1.0);
  return X(idx) + imag * Y(idx);
}

class ResultsBuffer {
public:
  xacc::internal_compiler::qreg q_buffer;
  double opt_val;
  std::vector<double> opt_params;
};

using Handle = std::future<ResultsBuffer>;
ResultsBuffer sync(Handle &handle) { return handle.get(); }

void set_verbose(bool verbose);

class ObjectiveFunction;

template <typename... Args> class ArgTranslator {
public:
  std::function<std::tuple<Args...>(std::vector<double>)> t;
  ArgTranslator(std::function<std::tuple<Args...>(std::vector<double> x)> &&ts)
      : t(ts) {}

  std::tuple<Args...> operator()(std::vector<double> x) { return t(x); }
};

template <typename... Args>
using TranslationFunctor =
    std::function<std::tuple<Args...>(const std::vector<double>)>;

namespace __internal__ {

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
  return f->operator()(std::get<I>(t)...);
}

template <typename Function, typename Tuple> auto call(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std::make_index_sequence<size>{});
}

// Given a quantum kernel functor / function pointer, create the xacc
// CompositeInstruction representation of it
template <typename QuantumKernel, typename... Args>
std::shared_ptr<CompositeInstruction>
kernel_as_composite_instruction(QuantumKernel &k, Args... args) {
  // turn off execution
  xacc::internal_compiler::__execute = false;
  // Execute to compile, this will store and we can get it
  k(args...);
  // turn execution on
  xacc::internal_compiler::__execute = true;
#ifdef QCOR_USE_QRT
  return quantum::getProgram();
#else
  return xacc::internal_compiler::getLastCompiled();
#endif
}

// Observe the given kernel, and return the expected value
double observe(std::shared_ptr<CompositeInstruction> program,
               std::shared_ptr<Observable> obs,
               xacc::internal_compiler::qreg &q);

// Observe the kernel and return the measured kernels
std::vector<std::shared_ptr<CompositeInstruction>>
observe(std::shared_ptr<Observable> obs,
        std::shared_ptr<CompositeInstruction> program);

// Get the objective function from the service registry
std::shared_ptr<ObjectiveFunction> get_objective(const char *type);

} // namespace __internal__

// The ObjectiveFunction represents a functor-like data structure that
// models a general parameterized scalar function. It is initialized with a
// problem-specific Observable and Quantum Kernel, and exposes a method for
// evaluation, given a list or array of scalar parameters.
// Implementations of this concept are problem-specific, and leverage the
// observe() functionality of the provided Observable to produce one or many
// measured Kernels that are then queued for execution on the available quantum
// co-processor, given the current value of the input parameters. The results of
// these quantum executions are to be used by the ObjectiveFunction to return a
// list of scalar values, representing the evaluation of the ObjectiveFunction
// at the given set of input parameters. Furthermore, the ObjectiveFunction has
// access to a global ResultBuffer that it uses to publish execution results at
// the current input parameters
class ObjectiveFunction : public xacc::Identifiable {
private:
  // This points to provided functor representation
  // of the quantum kernel, used to reconstruct
  // CompositeInstruction in variadic operator()
  void *pointer_to_functor = nullptr;

protected:
  // Pointer to the problem-specific Observable
  std::shared_ptr<Observable> observable;

  // Pointer to the quantum kernel
  std::shared_ptr<CompositeInstruction> kernel;

  // The buffer containing all execution results
  xacc::internal_compiler::qreg qreg;

  HeterogeneousMap options;

  // To be implemented by subclasses. Subclasses
  // can assume that the kernel has been evaluated
  // at current iterates (or evaluation) of the
  // objective function. I.e. this is called in
  // the variadic operator()(Args... args) method after
  // kernel->updateRuntimeArguments(args...)
  virtual double operator()() = 0;

public:
  // Publicly visible to clients for use in Optimization
  std::vector<double> current_gradient;

  // The Constructor
  ObjectiveFunction() = default;

  // Initialize this ObjectiveFunction with the problem
  // specific observable and CompositeInstruction
  virtual void initialize(std::shared_ptr<Observable> obs,
                          std::shared_ptr<CompositeInstruction> qk) {
    observable = obs;
    kernel = qk;
  }

  // Initialize this ObjectiveFunction with the problem
  // specific observable and pointer to quantum functor
  virtual void initialize(std::shared_ptr<Observable> obs, void *qk) {
    observable = obs;
    pointer_to_functor = qk;
  }

  void set_options(HeterogeneousMap &opts) { options = opts; }

  // Set the results buffer
  void set_qreg(xacc::internal_compiler::qreg q) { qreg = q; }
  xacc::internal_compiler::qreg get_qreg() { return qreg; }

  // Evaluate this Objective function at the give parameters.
  // These variadic parameters must mirror the provided
  // quantum kernel
  template <typename... ArgumentTypes>
  double operator()(ArgumentTypes... args) {
#ifdef QCOR_USE_QRT
    auto functor =
        reinterpret_cast<void (*)(ArgumentTypes...)>(pointer_to_functor);
    kernel = __internal__::kernel_as_composite_instruction(functor, args...);
#else
    if (!kernel) {
      auto functor =
          reinterpret_cast<void (*)(ArgumentTypes...)>(pointer_to_functor);
      kernel = __internal__::kernel_as_composite_instruction(functor, args...);
    }
#endif

    if (!qreg.results()) {
      // this hasn't been set, so set it
      qreg = std::get<0>(std::forward_as_tuple(args...));
    }

#ifndef QCOR_USE_QRT
    kernel->updateRuntimeArguments(args...);
#endif
    return operator()();
  }
};

void set_backend(const std::string &backend) {
  xacc::internal_compiler::compiler_InitializeXACC();
  xacc::internal_compiler::setAccelerator(backend.c_str());
}

std::shared_ptr<CompositeInstruction> compile(const std::string &src);

// Public observe function, returns expected value of Observable
template <typename QuantumKernel, typename... Args>
auto observe(QuantumKernel &kernel, std::shared_ptr<Observable> obs,
             Args... args) {
  auto program = __internal__::kernel_as_composite_instruction(kernel, args...);
  return [program, obs](Args... args) {
    // Get the first argument, which should be a qreg
    auto q = std::get<0>(std::forward_as_tuple(args...));

    // Set the arguments on the IR
#ifndef QCOR_USE_QRT
    program->updateRuntimeArguments(args...);
#endif

    // Observe the program
    auto programs = obs->observe(program);

    xacc::internal_compiler::execute(q.results(), programs);

    // We want to contract q children buffer
    // exp-val-zs with obs term coeffs
    return q.weighted_sum(obs.get());
  }(args...);
}

template <typename QuantumKernel, typename... Args>
auto observe(QuantumKernel &kernel, Observable &obs,
             Args... args) {
  auto program = __internal__::kernel_as_composite_instruction(kernel, args...);
  return [program, &obs](Args... args) {



    // Get the first argument, which should be a qreg
    auto q = std::get<0>(std::forward_as_tuple(args...));
    // std::cout << "\n" << program->toString() << "\n";

    // Set the arguments on the IR
#ifndef QCOR_USE_QRT
    program->updateRuntimeArguments(args...);
#endif

    // Observe the program
    auto programs = obs.observe(program);

    xacc::internal_compiler::execute(q.results(), programs);

    // We want to contract q children buffer
    // exp-val-zs with obs term coeffs
    return q.weighted_sum(&obs);
  }(args...);
}

// Create the desired Optimizer
std::shared_ptr<xacc::Optimizer>
createOptimizer(const char *type, HeterogeneousMap &&options = {});

// Create an observable from a string representation
std::shared_ptr<Observable> createObservable(const char *repr);

std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const char *obj_name, std::shared_ptr<CompositeInstruction> kernel,
    std::shared_ptr<Observable> observable, HeterogeneousMap &&options = {}) {
  auto obj_func = qcor::__internal__::get_objective(obj_name);
  obj_func->initialize(observable, kernel);
  obj_func->set_options(options);
  return obj_func;
}

// Create an Objective Function that makes calls to the
// provided Quantum Kernel, with measurements dictated by
// the provided Observable. Optionally can provide problem-specific
// options map.
template <typename QuantumKernel>
std::shared_ptr<ObjectiveFunction>
createObjectiveFunction(const char *obj_name, QuantumKernel &kernel,
                        std::shared_ptr<Observable> observable,
                        HeterogeneousMap &&options = {}) {
  auto obj_func = qcor::__internal__::get_objective(obj_name);
  // We can store this function pointer to a void* on ObjectiveFunction
  // to be converted to CompositeInstruction later
  void *kk = reinterpret_cast<void *>(kernel);
  obj_func->initialize(observable, kk);
  obj_func->set_options(options);
  return obj_func;
}

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    std::function<double(const std::vector<double>,
                                         std::vector<double> &)> &&opt_function,
                    const int nParameters);

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    qcor::OptFunction &&opt_function);

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    qcor::OptFunction &opt_function);
template <typename... Args>
Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    TranslationFunctor<Args...> translation,
                    const int nParameters) {
  return taskInitiate(
      objective, optimizer,
      [=](const std::vector<double> x, std::vector<double> &dx) {
        auto translated_tuple = translation(x);
        return qcor::__internal__::call(objective, translated_tuple);
      },
      nParameters);
}
} // namespace qcor

#endif
