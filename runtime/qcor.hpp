#ifndef RUNTIME_QCOR_HPP_
#define RUNTIME_QCOR_HPP_

#include <future>

#include "qpu_handler.hpp"

namespace xacc {
    class Function;
}

using namespace xacc;

namespace qcor {

using HandlerLambda = std::function<void(qpu_handler &)>;

// Persist the given function to file, return
// the file name
const std::string persistCompiledCircuit(std::shared_ptr<Function> function);

// Load the compiled circuit from file
std::shared_ptr<Function> loadCompiledCircuit(const std::string &fileName);

// Submit an asynchronous job to the QPU
std::future<int> submit(HandlerLambda &&);

} // namespace qcor

#endif
