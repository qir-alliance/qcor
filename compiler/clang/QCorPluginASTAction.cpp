#include "QCorPluginASTAction.hpp"
#include "QCorASTConsumer.hpp"

#include <iostream>

namespace qcor {
std::unique_ptr<ASTConsumer>
QCorPluginASTAction::CreateASTConsumer(CompilerInstance &ci, llvm::StringRef) {
  return llvm::make_unique<QCorASTConsumer>(ci);
}

bool QCorPluginASTAction::ParseArgs(const CompilerInstance &ci,
                                    const std::vector<std::string> &args) {
  if (!xacc::isInitialized()) {
    xacc::Initialize(args);
  }
  for (auto a : args)
    xacc::info("qcor argument: " + a);
  return true;
}

PluginASTAction::ActionType QCorPluginASTAction::getActionType() {
  return PluginASTAction::AddBeforeMainAction;
}

} // namespace qcor

static FrontendPluginRegistry::Add<qcor::QCorPluginASTAction>
    X("enable-quantum", "Enable quantum language extension via XACC.");