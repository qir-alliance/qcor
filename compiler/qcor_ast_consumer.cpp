#include "qcor_ast_consumer.hpp"
#include "qcor_ast_visitor.hpp"
#include <chrono>

using namespace clang;

namespace qcor {
namespace compiler {
QCORASTConsumer::QCORASTConsumer(CompilerInstance &c, Rewriter &rw)
    : ci(c),
      rewriter(rw) {}

bool QCORASTConsumer::HandleTopLevelDecl(DeclGroupRef DR) {
  QCORASTVisitor visitor(ci, rewriter);
  for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
      visitor.TraverseDecl(*b);
  }
  return true;
}
} // namespace compiler
} // namespace qcor