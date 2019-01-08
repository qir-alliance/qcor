#ifndef COMPILER_QUANTUMKERNELHANDLER_HPP_
#define COMPILER_QUANTUMKERNELHANDLER_HPP_

#include "clang/AST/ASTContext.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include "IRProvider.hpp"
#include "XACC.hpp"

#include <iostream>

using namespace clang;

namespace qcor {

class QuantumKernelHandler : public ExternalSemaSource {
public:
  QuantumKernelHandler(ASTContext &context) : m_Context(context) {
    auto irProvider = xacc::getService<xacc::IRProvider>("gate");
    validInstructions = irProvider->getInstructions();
  }

  bool LookupUnqualified(clang::LookupResult &R, clang::Scope *S) override {
    DeclarationName Name = R.getLookupName();
    std::string unknownName = Name.getAsString();

    // If this is a valid quantum instruction, tell Clang its 
    // all gonna be ok, we got this...
    if (std::find(validInstructions.begin(), validInstructions.end(),
                  unknownName) != validInstructions.end()) {
          std::cout << S->isClassScope() << ", " << S->isFunctionScope() << "\n";

    //   auto fnp = S->getFnParent();
    //   fnp->dump();
    //   std::cout << "DECL " << fnp->getEntity()->getDeclKindName() << "\n";
    //   fnp->getEntity()->dumpLookups();
      
    //   R.dump();

      IdentifierInfo *II = Name.getAsIdentifierInfo();
      SourceLocation Loc = R.getNameLoc();
      VarDecl *Result =
          VarDecl::Create(m_Context, R.getSema().getFunctionLevelDeclContext(),
                          Loc, Loc, II, m_Context.DependentTy, 0, SC_None);

    FunctionDecl * d = FunctionDecl::Create(m_Context, R.getSema().getFunctionLevelDeclContext(),
                          Loc, Loc, II, m_Context.VoidTy, 0, SC_None );
                          
    std::cout << "HELLO QK:\n";
    Result->dump();
    d->dump();

      if (Result) {
        R.addDecl(Result);
        return true;
      } else {
        return false;
      }
    }
    return false;
  }

private:
  ASTContext &m_Context;
  std::vector<std::string> validInstructions;
};
} // namespace qcor
#endif