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
    xacc::Initialize(args);
  }
  for (auto a : args)
    xacc::info("qcor argument: " + a);
  return true;
}

PluginASTAction::ActionType QCORPluginAction::getActionType() {
  return PluginASTAction::AddBeforeMainAction;
}
}
} // namespace qcor

static FrontendPluginRegistry::Add<qcor::compiler::QCORPluginAction>
    X("enable-quantum", "Enable quantum language extension via XACC.");