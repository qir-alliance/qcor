#ifndef RUNTIME_QCOR_HPP_
#define RUNTIME_QCOR_HPP_

#include <qalloc>
#include <future>
#include <memory>

#include "Observable.hpp"
#include "Optimizer.hpp"

#include "xacc_internal_compiler.hpp"

// Forward declare xacc types to speed up compile times
namespace xacc {
class Algorithm;
} // namespace xacc

namespace qcor {
using Handle = std::future<xacc::AcceleratorBuffer *>;

void initialize();
void finalize();

namespace __internal__ {

// std::size_t param_counter = 0;

// // Helper function for creating a vector of doubles
// // from a variadic pack of doubles
// void constructInitialParameters(double *p) { param_counter = 0; }
// template <typename First, typename... Rest>
// void constructInitialParameters(First *p, First firstArg, Rest... rest) {
//   p[param_counter] = firstArg;
//   param_counter++;
//   constructInitialParameters(p, rest...);
// }


// Given a quantum kernel functor, create the xacc
// CompositeInstruction representation of it
template <typename QuantumKernel, typename... Args>
xacc::CompositeInstruction *kernel_as_composite_instruction(QuantumKernel &k,
                                                            Args... args) {
  // How to know how many qregs to create?
  // oh well just create one tmp one
  //   auto q = qalloc(1);
  // turn off execution
  xacc::internal_compiler::__execute = false;
  // Execute to compile, this will store and we can get it
  k(args...);
  // turn execution on
  xacc::internal_compiler::__execute = true;
  return xacc::internal_compiler::getLastCompiled();
}

std::vector<std::shared_ptr<xacc::CompositeInstruction>>
observe(std::shared_ptr<xacc::Observable> obs, xacc::CompositeInstruction* program);

} // namespace __internal__

template <typename QuantumKernel, typename... Args>
auto observe(QuantumKernel &kernel, std::shared_ptr<xacc::Observable> obs, Args... args) {
  auto program = __internal__::kernel_as_composite_instruction(kernel, args...);
  return [program, obs](Args... args) {

    // Get the first argument, which should be a qreg
    auto q = std::get<0>(std::forward_as_tuple(args...));
    // std::cout << "\n" << program->toString() << "\n";

    // Set the arguments on the IR
    program->updateRuntimeArguments(args...);
    // std::cout << "\n" << program->toString() << "\n";

    // Observe the program
    auto programs = __internal__::observe(obs, program);
    
    std::vector<xacc::CompositeInstruction *> ptrs;
    for (auto p : programs)
      ptrs.push_back(p.get());

    xacc::internal_compiler::execute(q.results(), ptrs);

    // We want to contract q children buffer
    // exp-val-zs with obs term coeffs
    return q.weighted_sum(obs.get());
  }(args...);
}

// Get the default Optimizer
xacc::Optimizer *getOptimizer();

// Get a pauli observable from a string representation
std::shared_ptr<xacc::Observable> getObservable(const char *repr);


} // namespace qcor

#endif


// Helper function for extracting Variant keys from the
// underlying resultant AcceleratorBuffer. We specialize
// this for certain types (double, std::vector<double>)
// template <typename T>
// T extract_results(xacc::AcceleratorBuffer *q, const char *key);


// // Execute the hybrid variational Algorithm with given name, providing
// // the required Optimizer and Observable.
// Handle execute_algorithm(const char *algorithm,
//                          xacc::CompositeInstruction *program,
//                          xacc::Optimizer *opt, xacc::Observable *obs,
//                          double *parameters);
// Given some objective function (like VQE) that takes
// a parameterized circuit and quantum observable dictating
// measurements on that circuit, asynchronously compute
// circuit parameters that are optimal with respect to the
// return value of the objective function, using the provided
// classical optimizer. Clients can provide the initial
// circuit parameters (the Args... variadic parameter pack)
// template <typename QuantumKernel, typename... Args>
// Handle taskInitiate(QuantumKernel &kernel, const char *objective,
//                     xacc::Optimizer *opt, xacc::Observable *obs, Args... args) {
//   auto program = __internal__::kernel_as_composite_instruction(kernel, args...);
//   double *parameters = new double[sizeof...(args)];
//   __internal__::constructInitialParameters(parameters, args...);
//   auto handle =
//       __internal__::execute_algorithm(objective, program, opt, obs, parameters);
//   return handle;
// }

// Helper function for extracting Variant keys from the
// underlying resultant AcceleratorBuffer. We specialize
// this for certain types (double, std::vector<double>)
// template <typename T>
// T extract_results(xacc::AcceleratorBuffer *q, const char *key);


// // Execute the hybrid variational Algorithm with given name, providing
// // the required Optimizer and Observable.
// Handle execute_algorithm(const char *algorithm,
//                          xacc::CompositeInstruction *program,
//                          xacc::Optimizer *opt, xacc::Observable *obs,
//                          double *parameters);
// Given some objective function (like VQE) that takes
// a parameterized circuit and quantum observable dictating
// measurements on that circuit, asynchronously compute
// circuit parameters that are optimal with respect to the
// return value of the objective function, using the provided
// classical optimizer. Clients can provide the initial
// circuit parameters (the Args... variadic parameter pack)
// template <typename QuantumKernel, typename... Args>
// Handle taskInitiate(QuantumKernel &kernel, const char *objective,
//                     xacc::Optimizer *opt, xacc::Observable *obs, Args... args) {
//   auto program = __internal__::kernel_as_composite_instruction(kernel, args...);
//   double *parameters = new double[sizeof...(args)];
//   __internal__::constructInitialParameters(parameters, args...);
//   auto handle =
//       __internal__::execute_algorithm(objective, program, opt, obs, parameters);
//   return handle;
// }