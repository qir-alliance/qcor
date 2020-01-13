#ifndef RUNTIME_QCOR2_HPP_
#define RUNTIME_QCOR2_HPP_

#include "qalloc.hpp"
#include "xacc_internal_compiler.hpp"
#include <future>
#include <vector>

namespace xacc {
class Optimizer;
class Observable;
class Algorithm;
class CompositeInstruction;
} // namespace xacc

namespace qcor {
xacc::Optimizer *getOptimizer();
xacc::Observable *getObservable(const char *repr);

template<typename T>
T extract_results(xacc::internal_compiler::qreg& q, const char * key);

void constructInitialParameters(std::vector<double>& p) {}
template <typename First, typename... Rest>
void constructInitialParameters(std::vector<First> &p, First firstArg,
                                Rest... rest) {

  p.push_back(firstArg);
  constructInitialParameters(p, rest...);
}

std::future<xacc::internal_compiler::qreg>
execute_algorithm(const char *algorithm, xacc::CompositeInstruction *program,
                  xacc::Optimizer *opt, xacc::Observable *obs, std::vector<double>& parameters);

template <typename QuantumKernel, typename... Args>
xacc::CompositeInstruction *kernel_as_composite_instruction(QuantumKernel &k,
                                                            Args... args) {
  // How to know how many qregs to create?
  // oh well just create one tmp one
  auto q = qalloc(1);
  // turn off execution
  xacc::internal_compiler::__execute = false;
  // Execute to compile, this will store and we can get it
  k(q, args...);
  // turn execution on
  xacc::internal_compiler::__execute = true;
  return xacc::internal_compiler::getLastCompiled();
}

template <typename QuantumKernel, typename... Args>
std::future<xacc::internal_compiler::qreg>
taskInitiateWithSyntax(QuantumKernel &kernel, const char *objective,
                       xacc::Optimizer *opt, xacc::Observable *obs,
                       Args... args) {
  auto program = kernel_as_composite_instruction(kernel, args...);
  std::vector<double> parameters;
  constructInitialParameters(parameters, args...);
  return execute_algorithm(objective, program, opt, obs, parameters);
}
}
#endif