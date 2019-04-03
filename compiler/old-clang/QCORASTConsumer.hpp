#ifndef COMPILER_QCORASTCONSUMER_HPP_
#define COMPILER_QCORASTCONSUMER_HPP_

#include "clang/AST/ASTConsumer.h"

#include "FuzzyParsingExternalSemaSource.hpp"

namespace clang {
  class CompilerInstance;
}
using namespace clang;

namespace qcor {
namespace compiler {
class QCORASTConsumer : public ASTConsumer {
public:
  QCORASTConsumer(CompilerInstance &c);

  bool HandleTopLevelDecl(DeclGroupRef DR) override;

private:
  CompilerInstance &ci;
  std::shared_ptr<FuzzyParsingExternalSemaSource> fuzzyParser;
};
} // namespace compiler
} // namespace qcor
#endif