#include "qcor.hpp"
#include "qpu_handler.hpp"

#include "AcceleratorBuffer.hpp"
#include "IRProvider.hpp"
#include "XACC.hpp"
#include <regex>

using namespace xacc;

namespace qcor {

void Initialize(int argc, char** argv) {
    xacc::Initialize(argc,argv);
}
void Initialize(std::vector<std::string> argv) {
    xacc::Initialize(argv);
}

const std::string persistCompiledCircuit(std::shared_ptr<Function> function) {
      std::function<char()> randChar = []() -> char {
    const char charset[] = "0123456789"
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                           "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[rand() % max_index];
  };

  auto generateRandomString = [&](const int length = 10) -> const std::string {
    std::string str(length, 0);
    std::generate_n(str.begin(), length, randChar);
    return str;
  };

  auto file_name = generateRandomString();
  auto persistedFunction = xacc::getCompiler("xacc-py")->translate("", function);
  persistedFunction = persistedFunction.substr(7,persistedFunction.length());
  xacc::appendCache(file_name, "compiled", InstructionParameter(persistedFunction), ".qcor_cache");
  return file_name;
}

std::shared_ptr<Function> loadCompiledCircuit(const std::string& fileName) {
  auto cache = xacc::getCache(fileName, ".qcor_cache");
  if (!cache.count("compiled")) {
    xacc::error("Invalid quantum compilation cache.");
  }

  auto compiled = cache["compiled"].as<std::string>();
  return xacc::getCompiler("xacc-py")->compile(compiled)->getKernels()[0];
}

std::future<std::shared_ptr<AcceleratorBuffer>> submit(HandlerLambda &&totalJob) {
  // Create the QPU Handler to pass to the given
  // Handler HandlerLambda
  qpu_handler handler;
  return std::async(std::launch::async, [&]() {
    totalJob(handler);
    return handler.getResults();
  });
}

std::shared_ptr<Optimizer> getOptimizer(const std::string& name) {
    return xacc::getService<qcor::Optimizer>(name);
}

} // namespace qcor