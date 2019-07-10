#ifndef COMPILER_QCOREXTERNALSEMASOURCE_HPP_
#define COMPILER_QCOREXTERNALSEMASOURCE_HPP_

#include "clang/AST/ASTContext.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/Lookup.h"

using namespace clang;

namespace qcor {
namespace compiler {
class QCORExternalSemaSource : public ExternalSemaSource {
protected:
  ASTContext *m_Context;
public:
  void setASTContext(ASTContext *context) {m_Context = context;}
  virtual void initialize() = 0;
  
};
} // namespace compiler
} // namespace qcor
#endif