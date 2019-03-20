#include "qcor.hpp"

#include "IRProvider.hpp"
#include "XACC.hpp"

using namespace xacc;

namespace qcor {

const std::string persistCompiledCircuit(std::shared_ptr<Function> function) {
  return "";
}

std::shared_ptr<Function> loadCompiledCircuit() {
  auto provider = xacc::getService<IRProvider>("gate");
  auto function = provider->createFunction("tmp", {});
  return function;
}

std::future<int> submit(HandlerLambda &&totalJob) {
  // Create the QPU Handler to pass to the given
  // Handler HandlerLambda
  qpu_handler handler;
  return std::async(std::launch::async, [&] {
    totalJob(handler);
    return 1;
  });
}

} // namespace qcor