#ifndef RUNTIME_QCOR_HPP_
#define RUNTIME_QCOR_HPP_

#include "optimizer.hpp"
#include <future>

#include "Observable.hpp"
#include "algorithm.hpp"

namespace xacc {
class Function;
class AcceleratorBuffer;
class Accelerator;
class Observable;
} // namespace xacc

using namespace xacc;

namespace qcor {

class qpu_handler;

extern std::map<std::string, InstructionParameter> runtimeMap;

void Initialize(int argc, char **argv);
void Initialize(std::vector<std::string> argv);

// Persist the given function to file, return
// the file name
const std::string persistCompiledCircuit(std::shared_ptr<Function> function,
                                         std::shared_ptr<Accelerator> acc);
std::shared_ptr<Function> loadCompiledCircuit(const std::string &fileName);

void storeRuntimeVariable(const std::string name, int param);
void storeRuntimeVariable(const std::string name, InstructionParameter &&param);
std::map<std::string, InstructionParameter> getRuntimeMap();

// Submit an asynchronous job to the QPU
using HandlerLambda = std::function<void(qpu_handler &)>;
std::future<std::shared_ptr<AcceleratorBuffer>> submit(HandlerLambda &&);

std::shared_ptr<Optimizer> getOptimizer(const std::string &name);
std::shared_ptr<Optimizer>
getOptimizer(const std::string &name,
             std::map<std::string, InstructionParameter> &&options);

std::shared_ptr<Observable> getObservable(const std::string &type,
                                          const std::string &representation);
std::shared_ptr<Observable> getObservable();
std::shared_ptr<Observable> getObservable(const std::string &representation);

std::shared_ptr<algorithm::Algorithm> getAlgorithm(const std::string name);
} // namespace qcor

#include "qpu_handler.hpp"

#endif
