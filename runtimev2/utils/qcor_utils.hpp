#pragma once
#include <functional>
#include <future>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include "CompositeInstruction.hpp"
#include "qrt.hpp"
#include "xacc_internal_compiler.hpp"

namespace qcor {
using CompositeInstruction = xacc::CompositeInstruction;
using HeterogeneousMap = xacc::HeterogeneousMap;
using qreg = xacc::internal_compiler::qreg;

class ResultsBuffer {
public:
  xacc::internal_compiler::qreg q_buffer;
  double opt_val;
  std::vector<double> opt_params;
};

using Handle = std::future<ResultsBuffer>;
ResultsBuffer sync(Handle &handle);

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

using GradientEvaluator =
    std::function<void(std::vector<double> x, std::vector<double> &dx)>;

namespace __internal__ {

// Given a quantum kernel functor / function pointer, create the xacc
// CompositeInstruction representation of it
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

std::shared_ptr<qcor::CompositeInstruction> create_composite(std::string name);

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
  return f->operator()(std::get<I>(t)...);
}

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

template <typename TupleType, typename FunctionType>
void tuple_for_each(
    TupleType &&, FunctionType,
    std::integral_constant<
        size_t, std::tuple_size<
                    typename std::remove_reference<TupleType>::type>::value>) {}

template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<
                       TupleType>::type>::value>::type>
void tuple_for_each(TupleType &&t, FunctionType f,
                    std::integral_constant<size_t, I>) {
  f(std::get<I>(t));
  __internal__::tuple_for_each(std::forward<TupleType>(t), f,
                               std::integral_constant<size_t, I + 1>());
}

template <typename TupleType, typename FunctionType>
void tuple_for_each(TupleType &&t, FunctionType f) {
  __internal__::tuple_for_each(std::forward<TupleType>(t), f,
                               std::integral_constant<size_t, 0>());
}
} // namespace __internal__

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


void set_backend(const std::string &backend);

std::shared_ptr<CompositeInstruction> compile(const std::string &src);
void set_verbose(bool verbose);
bool get_verbose();
void set_shots(const int shots);
void error(const std::string &msg);

} // namespace qcor