/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#pragma once

#include <argparse.hpp>
// #include <complex>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>
#include <Eigen/Dense>

#include "qcor_ir.hpp"

namespace xacc {
class IRProvider;
class IRTransformation;
class AcceleratorBuffer;
namespace internal_compiler {
  class qreg;
}
}  // namespace xacc

namespace qcor {

namespace constants {
static constexpr double pi = 3.141592653589793238;
}

namespace arg {
// Expose ArgumentParser for custom instantiation
using ArgumentParser = argparse::ArgumentParser;

// Default argument parser
ArgumentParser &get_parser();

// Parameter packing
// Call add_argument with variadic number of string arguments
template <typename... Targs>
argparse::Argument &add_argument(Targs... Fargs) {
  return get_parser().add_argument(Fargs...);
}

// Getter for options with default values.
//  @throws std::logic_error if there is no such option
//  @throws std::logic_error if the option has no value
//  @throws std::bad_any_cast if the option is not of type T
//
template <typename T = std::string>
T get_argument(std::string_view argument_name) {
  return get_parser().get<T>(argument_name);
}

// Main entry point for parsing command-line arguments using this
//  ArgumentParser
//  @throws std::runtime_error in case of any invalid argument
//
void parse_args(int argc, const char *const argv[]);
}  // namespace arg


template <typename T>
using PairList = std::vector<std::pair<T, T>>;
using HeterogeneousMap = xacc::HeterogeneousMap;
using IRTransformation = xacc::IRTransformation;
using IRProvider = xacc::IRProvider;
using qreg = xacc::internal_compiler::qreg;
using UnitaryMatrix = Eigen::MatrixXcd;
using DenseMatrix = Eigen::MatrixXcd;

// The ResultsBuffer is returned upon completion of
// the taskInitiate async call, it contains the buffer,
// the optimal value for the objective function, and the
// optimal parameters
// class ResultsBuffer {
//  public:
//   xacc::internal_compiler::qreg q_buffer;
//   double opt_val;
//   std::vector<double> opt_params;
//   double value;
// };

// A Handle is just a future on ResultsBuffer
// using Handle = std::future<ResultsBuffer>;

// Sync up a Handle
// ResultsBuffer sync(Handle &handle);

// Indicate we have an error with the given message.
// This should abort execution
void error(const std::string &msg);
std::vector<std::string> split(const std::string &str, char delimiter);

void persist_var_to_qreg(const std::string &key, double &val, qreg &q);
void persist_var_to_qreg(const std::string &key, int &val, qreg &q);

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

UnitaryMatrix map_composite_to_unitary_matrix(
    std::shared_ptr<CompositeInstruction> composite);

std::string translate(const std::string compiler,
                      std::shared_ptr<CompositeInstruction> program);

void append_plugin_path(const std::string path);

// Internal function for creating a CompositeInstruction, this lets us
// keep XACC out of the include headers here and put it in the cpp.
std::shared_ptr<qcor::CompositeInstruction> create_composite(std::string name);

// Return the CTRL-U CompositeInstruction generator
std::shared_ptr<qcor::CompositeInstruction> create_ctrl_u();
std::shared_ptr<qcor::CompositeInstruction> create_and_expand_ctrl_u(
    HeterogeneousMap &&m);

// Return the IR Transformation
std::shared_ptr<qcor::IRTransformation> get_transformation(
    const std::string &transform_type);

// return the IR Provider
std::shared_ptr<qcor::IRProvider> get_provider();

// Decompose the given unitary matrix with the specified decomposition
// algorithm.
std::shared_ptr<qcor::CompositeInstruction>
decompose_unitary(const std::string algorithm, UnitaryMatrix &mat,
                  qreg &buffer);

// Decompose the given unitary matrix with the specified decomposition algorithm
// and optimizer
// std::shared_ptr<qcor::CompositeInstruction>
// decompose_unitary(const std::string algorithm, UnitaryMatrix &mat, qreg &buffer,
//                   std::shared_ptr<xacc::Optimizer> optimizer);

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
std::vector<double> random_vector(const double l_range, const double r_range,
                   const std::size_t size);
                    
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

// constexpr std::complex<double> I{0.0, 1.0};
// // Note: cannot use std::complex_literal
// // because https://gcc.gnu.org/onlinedocs/gcc/Complex.html#Complex
// inline constexpr std::complex<double> operator"" _i(long double x) noexcept {
//   return {0., static_cast<double>(x)};
// }

#define qcor_expect(test_condition)                                \
  {                                                                \
    if (!(test_condition)) {                                       \
      std::stringstream ss;                                        \
      ss << __FILE__ << ":" << __LINE__ << ": Assertion failed: '" \
         << #test_condition << "'.";                               \
      error(ss.str());                                             \
    }                                                              \
  }
}  // namespace qcor