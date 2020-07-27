#pragma once

#include "CompositeInstruction.hpp"
#include "IRProvider.hpp"
#include "IRTransformation.hpp"
#include "qrt.hpp"
#include "xacc_internal_compiler.hpp"

#include <Eigen/Dense>
#include <functional>
#include <future>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

namespace qcor {

// Typedefs mapping xacc types to qcor types
using CompositeInstruction = xacc::CompositeInstruction;
using HeterogeneousMap = xacc::HeterogeneousMap;
using IRTransformation = xacc::IRTransformation;
using IRProvider = xacc::IRProvider;
using qreg = xacc::internal_compiler::qreg;
using UnitaryMatrix = Eigen::MatrixXcd;

// The ResultsBuffer is returned upon completion of
// the taskInitiate async call, it contains the buffer,
// the optimal value for the objective function, and the
// optimal parameters
class ResultsBuffer {
public:
  xacc::internal_compiler::qreg q_buffer;
  double opt_val;
  std::vector<double> opt_params;
};

// A Handle is just a future on ResultsBuffer
using Handle = std::future<ResultsBuffer>;

// Sync up a Handle
ResultsBuffer sync(Handle &handle);

// ArgTranslator takes a std function that maps a
// vector<double> argument to a tuple corresponding to
// the arguments for the quantum kernel in question
//
// FIXME provide example here
template <typename... Args> class ArgTranslator {
public:
  std::function<std::tuple<Args...>(std::vector<double>)> t;
  ArgTranslator(std::function<std::tuple<Args...>(std::vector<double> x)> &&ts)
      : t(ts) {}

  std::tuple<Args...> operator()(std::vector<double> x) { return t(x); }
};

// The TranslationFunctor maps vector<double> to a tuple of Args...
template <typename... Args>
using TranslationFunctor =
    std::function<std::tuple<Args...>(const std::vector<double>)>;

// The GradientEvaluator is user-provided function that sets the
// gradient dx for a given variational iteration at parameter set x
using GradientEvaluator =
    std::function<void(std::vector<double> x, std::vector<double> &dx)>;

namespace __internal__ {

// Given a quantum kernel functor / function pointer, create the xacc
// CompositeInstruction representation of it
//
// FIXME May not need this anymore
template <typename QuantumKernel, typename... Args>
std::shared_ptr<CompositeInstruction>
kernel_as_composite_instruction(QuantumKernel &k, Args... args) {
  quantum::clearProgram();
  // turn off execution
  const auto cached_exec = xacc::internal_compiler::__execute;
  xacc::internal_compiler::__execute = false;
  // Execute to compile, this will store and we can get it
  k(args...);
  // turn execution on
  xacc::internal_compiler::__execute = cached_exec;
  return quantum::getProgram();
}

// Internal function for creating a CompositeInstruction, this lets us
// keep XACC out of the include headers here and put it in the cpp.
std::shared_ptr<qcor::CompositeInstruction> create_composite(std::string name);
std::shared_ptr<qcor::CompositeInstruction> create_ctrl_u();
std::shared_ptr<qcor::IRTransformation>
get_transformation(const std::string &transform_type);
std::shared_ptr<qcor::IRProvider> get_provider();
std::shared_ptr<qcor::CompositeInstruction>
decompose_unitary(const std::string algorithm, UnitaryMatrix &mat, const std::string buffer_name);

// Utility for calling a Functor via mapping a tuple of Args to
// a sequence of Args...
template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
  return f->operator()(std::get<I>(t)...);
}

// Utility for calling a Functor via mapping a tuple of Args to
// a sequence of Args...
template <typename Function, typename Tuple> auto call(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std::make_index_sequence<size>{});
}

// This internal utility class enables the merging of all
// quantum kernel double or std::vector<double> parameters
// into a single std::vector<double> (these correspond to circuit
// rotation parameters)
class ConvertDoubleLikeToVectorDouble {
public:
  std::vector<double> &vec;
  ConvertDoubleLikeToVectorDouble(std::vector<double> &v) : vec(v) {}
  void operator()(std::vector<double> tuple_element_vec) {
    for (auto &e : tuple_element_vec) {
      vec.push_back(e);
    }
  }
  void operator()(double tuple_element_double) {
    vec.push_back(tuple_element_double);
  }
  template <typename T> void operator()(T &) {}
};

// Utility function for looping over tuple elements
template <typename TupleType, typename FunctionType>
void tuple_for_each(
    TupleType &&, FunctionType,
    std::integral_constant<
        size_t, std::tuple_size<
                    typename std::remove_reference<TupleType>::type>::value>) {}
// Utility function for looping over tuple elements
template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<
                       TupleType>::type>::value>::type>
// Utility function for looping over tuple elements
void tuple_for_each(TupleType &&t, FunctionType f,
                    std::integral_constant<size_t, I>) {
  f(std::get<I>(t));
  __internal__::tuple_for_each(std::forward<TupleType>(t), f,
                               std::integral_constant<size_t, I + 1>());
}
// Utility function for looping over tuple elements
template <typename TupleType, typename FunctionType>
void tuple_for_each(TupleType &&t, FunctionType f) {
  __internal__::tuple_for_each(std::forward<TupleType>(t), f,
                               std::integral_constant<size_t, 0>());
}
} // namespace __internal__

// Create and return a random vector<ScalarType> of the given size
// where all elements are within the given range
template <typename ScalarType>
auto random_vector(const ScalarType l_range, const ScalarType r_range,
                   const std::size_t size) {
  // Generate a random initial parameter set
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_real_distribution<ScalarType> dist{l_range, r_range};
  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
  std::vector<ScalarType> vec(size);
  std::generate(vec.begin(), vec.end(), gen);
  return vec;
}

// Take a function and return a TranslationFunctor
template <typename... Args>
auto args_translator(
    std::function<std::tuple<Args...>(const std::vector<double>)>
        &&functor_lambda) {
  return TranslationFunctor<Args...>(functor_lambda);
}

// C++17 python-like enumerate utility function
template <typename T, typename TIter = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))>
constexpr auto enumerate(T &&iterable) {
  struct iterator {
    size_t i;
    TIter iter;
    bool operator!=(const iterator &other) const { return iter != other.iter; }
    void operator++() {
      ++i;
      ++iter;
    }
    auto operator*() const { return std::tie(i, *iter); }
  };
  struct iterable_wrapper {
    T iterable;
    auto begin() { return iterator{0, std::begin(iterable)}; }
    auto end() { return iterator{0, std::end(iterable)}; }
  };
  return iterable_wrapper{std::forward<T>(iterable)};
}

// Set the backend that quantum kernels are targeting
void set_backend(const std::string &backend);

// Utility function to expose the XASM xacc Compiler
// from the cpp implementation and not include it in the header list above.
std::shared_ptr<CompositeInstruction> compile(const std::string &src);

// Toggle verbose mode
void set_verbose(bool verbose);

// Get if we are verbose or not
bool get_verbose();

// Set the shots for a given quantum kernel execution
void set_shots(const int shots);

// Indicate we have an error with the given message.
// This should abort execution
void error(const std::string &msg);

} // namespace qcor