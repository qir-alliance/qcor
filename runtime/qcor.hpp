#ifndef RUNTIME_QCOR_HPP_
#define RUNTIME_QCOR_HPP_

#include <future>

#include "optimizer.hpp"
#include "algorithm.hpp"

#include "Observable.hpp"

namespace xacc {
class Function;
class AcceleratorBuffer;
class Accelerator;
} // namespace xacc

using namespace xacc;

namespace qcor {

class qpu_handler;

using HandlerLambda = std::function<void(qpu_handler &)>;

extern std::map<std::string, InstructionParameter> runtimeMap;

void Initialize(int argc, char **argv);
void Initialize(std::vector<std::string> argv);

// Persist the given function to file, return
// the file name
const std::string persistCompiledCircuit(std::shared_ptr<Function> function,
                                         std::shared_ptr<Accelerator> acc);
std::shared_ptr<Function> loadCompiledCircuit(const std::string &fileName);

// void storeRuntimeVariable(const std::string name, int param);
void storeRuntimeVariable(const std::string name, InstructionParameter param);
std::map<std::string, InstructionParameter> getRuntimeMap();

// Submit an asynchronous job to the QPU
std::future<std::shared_ptr<AcceleratorBuffer>> submit(HandlerLambda &&lambda);
std::future<std::shared_ptr<AcceleratorBuffer>> submit(HandlerLambda &&lambda, std::shared_ptr<AcceleratorBuffer> buffer);

// std::shared_ptr<AcceleratorBuffer> submit(HandlerLambda &&lambda);

std::shared_ptr<Optimizer> getOptimizer(const std::string &name);
std::shared_ptr<Optimizer>
getOptimizer(const std::string &name,
             std::map<std::string, InstructionParameter> &&options);

std::shared_ptr<Observable> getObservable(const std::string &type,
                                          const std::string &representation);
std::shared_ptr<Observable> getObservable();
std::shared_ptr<Observable> getObservable(const std::string &representation);
std::shared_ptr<Observable> getObservable(const std::string &type, std::map<std::string, InstructionParameter> &&options);

std::shared_ptr<algorithm::Algorithm> getAlgorithm(const std::string name);
} // namespace qcor

#include "qpu_handler.hpp"

#endif
