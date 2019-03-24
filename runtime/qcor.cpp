#include "qcor.hpp"
#include "qpu_handler.hpp"

#include "AcceleratorBuffer.hpp"
#include "IRProvider.hpp"
#include "XACC.hpp"

using namespace xacc;

namespace qcor {

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



  return generateRandomString();
}

std::shared_ptr<Function> loadCompiledCircuit(const std::string& fileName) {
  auto provider = xacc::getService<IRProvider>("gate");
  auto function = provider->createFunction("tmp", {});
  auto h = provider->createInstruction("H", {0});
  auto cx = provider->createInstruction("CNOT", {0,1});
  auto m1 = provider->createInstruction("Measure", {0}, {InstructionParameter(0)});
  auto m2 = provider->createInstruction("Measure", {1}, {InstructionParameter(1)});

  function->addInstruction(h);
  function->addInstruction(cx);
  function->addInstruction(m1);
  function->addInstruction(m2);

  return function;
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

} // namespace qcor