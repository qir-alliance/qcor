#ifndef COMPILER_QCORPLUGINASTACTION_HPP_
#define COMPILER_QCORPLUGINASTACTION_HPP_

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace qcor {
namespace compiler {
class QCORPluginAction : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override;
  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override;
  PluginASTAction::ActionType getActionType() override;
};
}
} // namespace qcor

#endif