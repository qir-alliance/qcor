#ifndef COMPILER_CLANG_QCORPRAGMAHANDLER_HPP__
#define COMPILER_CLANG_QCORPRAGMAHANDLER_HPP__
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/TokenKinds.def"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include <iostream>
#include <sstream>

using namespace clang;

namespace qcor {
namespace compiler {
class QCORPragmaHandler : public PragmaHandler {
protected:
  Rewriter* rewriter;
public:
  QCORPragmaHandler() : PragmaHandler("qcor") {}
  void setRewriter(Rewriter* r) { rewriter = r; }
};

} // namespace compiler
} // namespace qcor

#endif