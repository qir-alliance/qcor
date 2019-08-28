#ifndef COMPILER_QCORASTCONSUMER_HPP_
#define COMPILER_QCORASTCONSUMER_HPP_

#include "clang/AST/ASTConsumer.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace clang {
class CompilerInstance;
}
using namespace clang;

namespace qcor {
namespace compiler {
class QCORASTConsumer : public ASTConsumer {
public:
  QCORASTConsumer(CompilerInstance &c, Rewriter &rw);

  bool HandleTopLevelDecl(DeclGroupRef DR) override;

private:
  CompilerInstance &ci;
  Rewriter &rewriter;
};
} // namespace compiler
} // namespace qcor
#endif