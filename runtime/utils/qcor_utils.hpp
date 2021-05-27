#pragma once

#include <argparse.hpp>
#include <Eigen/Dense>
#include <functional>
#include <future>
#include <memory>
#include <random>
#include <tuple>
#include <vector>
#include "AllGateVisitor.hpp"
#include "CompositeInstruction.hpp"
#include "IRProvider.hpp"
#include "IRTransformation.hpp"
#include "Optimizer.hpp"
#include "qalloc.hpp"
#include "xacc_internal_compiler.hpp"

namespace qcor {

namespace constants {
static constexpr double pi = 3.141592653589793238;
}

// Expose argparse
using ArgumentParser = argparse::ArgumentParser;

// Typedefs mapping xacc/Eigen types to qcor types
using CompositeInstruction = xacc::CompositeInstruction;
using HeterogeneousMap = xacc::HeterogeneousMap;
using IRTransformation = xacc::IRTransformation;
using IRProvider = xacc::IRProvider;
using qreg = xacc::internal_compiler::qreg;
using UnitaryMatrix = Eigen::MatrixXcd;
using DenseMatrix = Eigen::MatrixXcd;
using DenseVector = Eigen::VectorXcd;

template <typename T>
using PairList = std::vector<std::pair<T, T>>;

// The ResultsBuffer is returned upon completion of
// the taskInitiate async call, it contains the buffer,
// the optimal value for the objective function, and the
// optimal parameters
class ResultsBuffer {
 public:
  xacc::internal_compiler::qreg q_buffer;
  double opt_val;
  std::vector<double> opt_params;
  double value;
};

// A Handle is just a future on ResultsBuffer
using Handle = std::future<ResultsBuffer>;

// Sync up a Handle
ResultsBuffer sync(Handle &handle);

// Indicate we have an error with the given message.
// This should abort execution
void error(const std::string &msg);

template <typename T>
std::vector<T> linspace(T a, T b, size_t N) {
  T h = (b - a) / static_cast<T>(N - 1);
  std::vector<T> xs(N);
  typename std::vector<T>::iterator x;
  T val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) *x = val;
  return xs;
}

inline std::vector<int> range(int N) {
  std::vector<int> vec(N);
  std::iota(vec.begin(), vec.end(), 0);
  return vec;
}

inline std::vector<int> range(int start, int stop, int step) {
  if (step == 0) {
    error("step for range must be non-zero.");
  }

  int i = start;
  std::vector<int> vec;
  while ((step > 0) ? (i < stop) : (i > stop)) {
    vec.push_back(i);
    i += step;
  }
  return vec;
}

inline std::vector<int> range(int start, int stop) {
  return range(start, stop, 1);
}

// Get size() of any types that have size() implemented.
template <typename T>
int len(const T &countable) {
  return countable.size();
}
template <typename T>
int len(T &countable) {
  return countable.size();
}

// Python-like print instructions:
inline void print() { std::cout << "\n"; }
template <typename T, typename... TAIL>
void print(const T &t, TAIL... tail) {
  std::cout << t << " ";
  print(tail...);
}

template <typename T>
void persist_var_to_qreq(const std::string &key, T &val, qreg &q) {
  q.results()->addExtraInfo(key, val);
}

std::shared_ptr<qcor::IRTransformation> createTransformation(
    const std::string &transform_type);
    
// The TranslationFunctor maps vector<double> to a tuple of Args...
template <typename... Args>
using TranslationFunctor =
    std::function<std::tuple<Args...>(const std::vector<double> &)>;

// ArgsTranslator takes a std function that maps a
// vector<double> argument to a tuple corresponding to
// the arguments for the quantum kernel in question
//
// FIXME provide example here
template <typename... Args>
class ArgsTranslator {
 protected:
  TranslationFunctor<Args...> functor;

 public:
  ArgsTranslator(TranslationFunctor<Args...> ts) : functor(ts) {}

  std::tuple<Args...> operator()(const std::vector<double> &x) {
    return functor(x);
  }
};

template <typename... Args>
std::shared_ptr<ArgsTranslator<Args...>> createArgsTranslator(
    TranslationFunctor<Args...> functor) {
  return std::make_shared<ArgsTranslator<Args...>>(functor);
}

// The GradientEvaluator is user-provided function that sets the
// gradient dx for a given variational iteration at parameter set x
using GradientEvaluator =
    std::function<void(std::vector<double> x, std::vector<double> &dx)>;

namespace __internal__ {

std::string translate(const std::string compiler, std::shared_ptr<CompositeInstruction> program);

void append_plugin_path(const std::string path);

// Internal function for creating a CompositeInstruction, this lets us
// keep XACC out of the include headers here and put it in the cpp.
std::shared_ptr<qcor::CompositeInstruction> create_composite(std::string name);

// Return the CTRL-U CompositeInstruction generator
std::shared_ptr<qcor::CompositeInstruction> create_ctrl_u();

// Return the IR Transformation
std::shared_ptr<qcor::IRTransformation> get_transformation(
    const std::string &transform_type);

// return the IR Provider
std::shared_ptr<qcor::IRProvider> get_provider();

// Decompose the given unitary matrix with the specified decomposition
// algorithm.
std::shared_ptr<qcor::CompositeInstruction> decompose_unitary(
    const std::string algorithm, UnitaryMatrix &mat,
    const std::string buffer_name);

// Decompose the given unitary matrix with the specified decomposition algorithm
// and optimizer
std::shared_ptr<qcor::CompositeInstruction> decompose_unitary(
    const std::string algorithm, UnitaryMatrix &mat,
    const std::string buffer_name, std::shared_ptr<xacc::Optimizer> optimizer);

// Utility for calling a Functor via mapping a tuple of Args to
// a sequence of Args...
template <typename Function, typename Tuple, size_t... I>
auto evaluate_shared_ptr_fn_with_tuple_args(Function f, Tuple t,
                                            std::index_sequence<I...>) {
  return f->operator()(std::get<I>(t)...);
}

// Utility for calling a Functor via mapping a tuple of Args to
// a sequence of Args...
template <typename Function, typename Tuple>
auto evaluate_shared_ptr_fn_with_tuple_args(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return evaluate_shared_ptr_fn_with_tuple_args(
      f, t, std::make_index_sequence<size>{});
}

// Utility for calling a Functor via mapping a tuple of Args to
// a sequence of Args...
template <typename Function, typename Tuple, size_t... I>
auto evaluate_function_with_tuple_args(Function f, Tuple t,
                                       std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

// Utility for calling a Functor via mapping a tuple of Args to
// a sequence of Args...
template <typename Function, typename Tuple>
auto evaluate_function_with_tuple_args(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return evaluate_function_with_tuple_args(f, t,
                                           std::make_index_sequence<size>{});
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
  template <typename T>
  void operator()(T &) {}
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
}  // namespace __internal__

// Create and return a random vector<ScalarType> of the given size
// where all elements are within the given range
template <typename ScalarType>
auto random_vector(const ScalarType l_range, const ScalarType r_range,
                   const std::size_t size) {
  // Generate a random initial parameter set
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};  // Generates random integers
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

using namespace Eigen;

// The following visitor is adapted from xacc AutodiffVisitor.

constexpr std::complex<double> I{0.0, 1.0};
// Note: cannot use std::complex_literal
// because https://gcc.gnu.org/onlinedocs/gcc/Complex.html#Complex
inline constexpr std::complex<double> operator"" _i(long double x) noexcept {
  return {0., static_cast<double>(x)};
}

MatrixXcd kroneckerProduct(MatrixXcd &lhs, MatrixXcd &rhs);
enum class Rot { X, Y, Z };

class KernelToUnitaryVisitor : public xacc::quantum::AllGateVisitor {
 public:
  KernelToUnitaryVisitor(size_t in_nbQubits);

  void visit(xacc::quantum::Hadamard &h) override;
  void visit(xacc::quantum::CNOT &cnot) override;
  void visit(xacc::quantum::Rz &rz) override;
  void visit(xacc::quantum::Ry &ry) override;
  void visit(xacc::quantum::Rx &rx) override;
  void visit(xacc::quantum::X &x) override;
  void visit(xacc::quantum::Y &y) override;
  void visit(xacc::quantum::Z &z) override;
  void visit(xacc::quantum::CY &cy) override;
  void visit(xacc::quantum::CZ &cz) override;
  void visit(xacc::quantum::Swap &s) override;
  void visit(xacc::quantum::CRZ &crz) override;
  void visit(xacc::quantum::CH &ch) override;
  void visit(xacc::quantum::S &s) override;
  void visit(xacc::quantum::Sdg &sdg) override;
  void visit(xacc::quantum::T &t) override;
  void visit(xacc::quantum::Tdg &tdg) override;
  void visit(xacc::quantum::CPhase &cphase) override;
  void visit(xacc::quantum::Measure &measure) override;
  void visit(xacc::quantum::Identity &i) override;
  void visit(xacc::quantum::U &u) override;
  void visit(xacc::quantum::IfStmt &ifStmt) override;
  // Identifiable Impl
  const std::string name() const override { return "kernel-to-unitary"; }
  const std::string description() const override { return ""; }
  MatrixXcd getMat() const;
  MatrixXcd singleQubitGateExpand(MatrixXcd &in_gateMat, size_t in_loc) const;

  MatrixXcd singleParametricQubitGateExpand(double in_var, Rot in_rotType,
                                            size_t in_bitLoc) const;

  MatrixXcd twoQubitGateExpand(MatrixXcd &in_gateMat, size_t in_bit1,
                               size_t in_bit2) const;

 private:
  MatrixXcd m_circuitMat;
  size_t m_nbQubit;
};

#define qcor_expect(test_condition)                                            \
  {                                                                            \
    if (!(test_condition)) {                                                   \
      std::stringstream ss;                                                    \
      ss << __FILE__ << ":" << __LINE__ << ": Assertion failed: '"             \
         << #test_condition << "'.";                                           \
      error(ss.str());                                                         \
    }                                                                          \
  }
} // namespace qcor