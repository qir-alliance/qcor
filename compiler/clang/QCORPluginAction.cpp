#include "QCORPluginAction.hpp"
#include "QCORASTConsumer.hpp"

#include "XACC.hpp"

namespace qcor {
namespace compiler {
std::unique_ptr<ASTConsumer>
QCORPluginAction::CreateASTConsumer(CompilerInstance &ci, llvm::StringRef) {
  return llvm::make_unique<QCORASTConsumer>(ci);
}

bool QCORPluginAction::ParseArgs(const CompilerInstance &ci,
                                 const std::vector<std::string> &args) {
  if (!xacc::isInitialized()) {
    std::vector<std::string> local;
    local.push_back("--logger-name");
    local.push_back("qcor");

    xacc::Initialize(local);
  }
//   for (auto a : args) {
//     xacc::info("qcor argument: " + a);
//   }

  auto it = std::find(args.begin(), args.end(), "accelerator");
  if (it != args.end()) {
    int index = std::distance(args.begin(), it);
    auto acc = args[index + 1];
    xacc::setAccelerator(acc);
  }

  std::vector<std::string> transformations;
  it = args.begin();
  std::for_each(args.begin(), args.end(), [&](const std::string &value) {
    if (value == "transform") {
      int index = std::distance(args.begin(), it);
      auto transformationName = args[index + 1];
      transformations.push_back(transformationName);
    }
    ++it;
  });

  if (!transformations.empty()) {
    std::string transformNames = transformations[0];
    for (int i = 1; i < transformations.size(); ++i) {
      transformNames += "," + transformations[i];
    }
    xacc::setOption("qcor-transforms",transformNames);
  }
  return true;
}

PluginASTAction::ActionType QCORPluginAction::getActionType() {
  return PluginASTAction::AddBeforeMainAction;
}
} // namespace compiler
} // namespace qcor

static FrontendPluginRegistry::Add<qcor::compiler::QCORPluginAction>
    X("enable-quantum", "Enable quantum language extension via XACC.");