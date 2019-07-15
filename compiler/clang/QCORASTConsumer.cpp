#include "QCORASTConsumer.hpp"
#include "LambdaVisitor.hpp"
#include <chrono>

// #include "clang/ASTMatchers/ASTMatchFinder.h"
// #include "clang/ASTMatchers/ASTMatchers.h"

// using namespace clang::ast_matchers;

using namespace clang;

namespace qcor {
namespace compiler {
QCORASTConsumer::QCORASTConsumer(CompilerInstance &c, Rewriter &rw)
    : ci(c),
      rewriter(rw) {}

bool QCORASTConsumer::HandleTopLevelDecl(DeclGroupRef DR) {
  LambdaVisitor visitor(ci, rewriter);
  for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
      visitor.TraverseDecl(*b);
  }
  return true;
}
} // namespace compiler
} // namespace qcor