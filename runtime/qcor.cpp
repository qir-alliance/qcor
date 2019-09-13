#include "qcor.hpp"

#include "AcceleratorBuffer.hpp"
#include "IRProvider.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "xacc_observable.hpp"

// #include "PauliOperator.hpp"
#include "CountGatesOfTypeVisitor.hpp"
#include "CommonGates.hpp"

#include <regex>

using namespace xacc;

namespace qcor {

namespace __internal {
bool executeKernel = true;
void switchDefaultKernelExecution(bool execute) { executeKernel = execute; }
bool hasMeasurements(std::shared_ptr<CompositeInstruction> inst) {
  quantum::CountGatesOfTypeVisitor<quantum::Measure> count(inst);
  return count.countGates() > 0;
}

void updateMap(xacc::HeterogeneousMap &m, std::vector<double> &values) {
  if (values.empty()) {
    m.get_mutable<std::vector<double>>("initial-parameters") =
        std::vector<double>(m.get<std::size_t>("__internal_n_vars"));
  } else {
    m.get_mutable<std::vector<double>>("initial-parameters") = values;
  }
}

void updateMap(xacc::HeterogeneousMap &m, std::vector<double> &&values) {
  if (values.empty()) {
    m.get_mutable<std::vector<double>>("initial-parameters") =
        std::vector<double>(m.get<std::size_t>("__internal_n_vars"));
  } else {
    m.get_mutable<std::vector<double>>("initial-parameters") = values;
  }
}

void updateMap(xacc::HeterogeneousMap &m, double value) {
  m.get_mutable<std::vector<double>>("initial-parameters").push_back(value);
}

void constructInitialParameters(xacc::HeterogeneousMap &m) { return; }
} // namespace __internal

void Initialize() { Initialize(std::vector<std::string>{}); }

void Initialize(int argc, char **argv) {
  std::vector<const char *> tmp(argv, argv + argc);
  std::vector<std::string> newargv;
  for (auto &t : tmp)
    newargv.push_back(std::string(t));
  Initialize(newargv);
}

void Initialize(std::vector<std::string> argv) {
  argv.push_back("--logger-name");
  argv.push_back("qcor");
  xacc::Initialize(argv);
}
void Finalize() { xacc::Finalize(); }

ResultBuffer qalloc(const std::size_t nBits) { return xacc::qalloc(nBits); }
ResultBuffer qalloc() { return xacc::qalloc(); }
ResultBuffer sync(Handle &handle) { return handle.get(); }

Handle submit(HandlerLambda &&totalJob) {
  // Create the QPU Handler to pass to the given
  // Handler HandlerLambda
  return std::async(std::launch::async, [=]() { // bug must be by value...
    qpu_handler handler;
    totalJob(handler);
    return handler.getResults();
  });
}

Handle submit(HandlerLambda &&totalJob,
              std::shared_ptr<AcceleratorBuffer> buffer) {
  return std::async(std::launch::async, [&]() {
    qpu_handler handler(buffer);
    totalJob(handler);
    return handler.getResults();
  });
}

std::shared_ptr<Optimizer> getOptimizer(const std::string &name) {
  return xacc::getOptimizer(name);
}
std::shared_ptr<Optimizer> getOptimizer(const std::string &name,
                                        const HeterogeneousMap &&options) {
  return xacc::getOptimizer(name, options);
}

std::shared_ptr<Observable> getObservable(const std::string &type,
                                          const std::string &representation) {
  return xacc::quantum::getObservable(type, representation);
}

std::shared_ptr<Observable> getObservable() {
  return xacc::quantum::getObservable();
}
std::shared_ptr<Observable> getObservable(const std::string &representation) {
  return xacc::quantum::getObservable("pauli", representation);
}

std::shared_ptr<Observable> getObservable(const std::string &type,
                                          const HeterogeneousMap &&options) {
  return xacc::quantum::getObservable(type, options);
}

} // namespace qcor
