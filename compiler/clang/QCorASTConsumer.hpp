#ifndef COMPILER_QCORASTCONSUMER_HPP_
#define COMPILER_QCORASTCONSUMER_HPP_

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"

#include "QcorASTVisitor.hpp"
#include "QuantumKernelHandler.hpp"

using namespace clang;

namespace qcor {
class QCorASTConsumer : public ASTConsumer {
public:
  QCorASTConsumer(CompilerInstance &c)
      : ci(c),
        handler(std::make_shared<QuantumKernelHandler>(c.getASTContext())) {}

  // Override the method that gets called for each parsed top-level
  // declaration.
  bool HandleTopLevelDecl(DeclGroupRef DR) override {
    QcorASTVisitor Visitor(ci);
    ci.getSema().addExternalSource(handler.get());
    // Visitor.TraverseDecl(ci.getASTContext().getTranslationUnitDecl());
    for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
      // Traverse the declaration using our AST visitor.
      Visitor.TraverseDecl(*b);
    }
    // (*DR.begin())->dump();
    return true;
  }

private:
  CompilerInstance &ci;
  std::shared_ptr<QuantumKernelHandler> handler;
};
} // namespace qcor
#endif