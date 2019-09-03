#include "fuzzy_parsing.hpp"

#include "IRProvider.hpp"
#include "XACC.hpp"
#include "xacc_service.hpp"

#include "Utils.hpp"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTImporter.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include "qcor_clang_utils.hpp"

using namespace clang;
using namespace clang::ast_matchers;

namespace qcor {
namespace compiler {

void FuzzyParsingExternalSemaSource::initialize() {
  auto provider = xacc::getService<xacc::IRProvider>("quantum");
  validInstructions = provider->getInstructions();
  validInstructions.push_back("CX");

  std::vector<std::shared_ptr<xacc::Instruction>> composites;
  std::string totalTempSource = "";
  for (auto &instructionName : validInstructions) {
    std::string tmpSource = "void " + instructionName + "(";
    auto tmpInst = provider->createInstruction(
        instructionName == "CX" ? "CNOT" : instructionName, {});
    if (!tmpInst->isComposite()) {
      int nRequiredBits = tmpInst->nRequiredBits();
      tmpSource += "int q0";
      for (int i = 1; i < nRequiredBits; i++) {
        tmpSource += ", int q" + std::to_string(i);
      }
      if (tmpInst->isParameterized() && instructionName != "Measure") {
        int nRequiredParams = tmpInst->nParameters();
        tmpSource += ", double p0";
        for (int i = 1; i < nRequiredParams; i++) {
          tmpSource += ", double p" + std::to_string(i);
        }
      }
      tmpSource += "){return;}";
      quantumInstruction2src.insert(
          {instructionName + "__qcor_instruction", tmpSource});
    } else {
      compositeInstructions.push_back(instructionName + "__qcor_instruction");
    }
  }

  hast = tooling::buildASTFromCodeWithArgs(
      "#include \"heterogeneous.hpp\"\nvoid f(xacc::HeterogeneousMap&& "
      "m, std::vector<double>& x){return;}",
      {"-std=c++14", "-I/home/cades/.xacc/include/xacc"});
  hMapRValue = FirstDeclMatcher<ParmVarDecl>().match(
      hast->getASTContext().getTranslationUnitDecl(), namedDecl(hasName("m")));
  stdVector = FirstDeclMatcher<ParmVarDecl>().match(
      hast->getASTContext().getTranslationUnitDecl(), namedDecl(hasName("x")));
}

bool FuzzyParsingExternalSemaSource::LookupUnqualified(clang::LookupResult &R,
                                                       clang::Scope *S) {
  std::string unknownName = R.getLookupName().getAsString();

  // If this is a valid quantum instruction, tell Clang not to error
  if (quantumInstruction2src.count(unknownName + "__qcor_instruction") &&
      S->getFlags() != 128 && S->getBlockParent() != nullptr) {

    auto Matcher = namedDecl(hasName(unknownName));

    auto ast = tooling::buildASTFromCodeWithArgs(
        quantumInstruction2src[unknownName + "__qcor_instruction"],
        {"-std=c++11"});
    FunctionDecl *D0 = FirstDeclMatcher<FunctionDecl>().match(
        ast->getASTContext().getTranslationUnitDecl(), Matcher);

    quantumInstructionASTs.push_back(std::move(ast));

    R.addDecl(D0);
    // D0->dump();
  } else if (std::find(compositeInstructions.begin(),
                       compositeInstructions.end(),
                       unknownName + "__qcor_instruction") !=
             std::end(compositeInstructions)) {

    if (!qbit) {
      // Save pointers to xacc::qbit, xacc::HeterogeneousMap&& ParmVarDecl
      qbit = FirstDeclMatcher<ParmVarDecl>().match(
          ci.getASTContext().getTranslationUnitDecl(),
          parmVarDecl(hasType(recordDecl(matchesName("xacc::qbit")))));
    }

    // This is a Circuit Generator CompositeInstruction. We assume
    // (for now) it has must have prototype
    // f(qbit q, std::vector<double>& x, HeterogeneousMap&&)
    // or f(qbit q, HeterogeneousMap&&)

    // Create a new ParmVarDecl exactly like that one
    auto qb_copy = ParmVarDecl::Create(
        ci.getASTContext(), ci.getSema().getFunctionLevelDeclContext(),
        SourceLocation(), SourceLocation(), qbit->getIdentifier(),
        qbit->getType(), 0, SC_None, nullptr);
   auto v_copy = ParmVarDecl::Create(
        ci.getASTContext(), ci.getSema().getFunctionLevelDeclContext(),
        SourceLocation(), SourceLocation(), stdVector->getIdentifier(),
        stdVector->getType(), 0, SC_None, nullptr);
    auto h_copy = ParmVarDecl::Create(
        ci.getASTContext(), ci.getSema().getFunctionLevelDeclContext(),
        SourceLocation(), SourceLocation(), hMapRValue->getIdentifier(),
        hMapRValue->getType(), 0, SC_None, nullptr);

    // Use astContext.getFunctionType (RETURNTYPE, ARGSPARMVARS, fpi)
    // to create a new Function QualType
    std::vector<QualType> ParamTypes;
    ParamTypes.push_back(qb_copy->getType());
    ParamTypes.push_back(v_copy->getType());
    ParamTypes.push_back(h_copy->getType());
    FunctionProtoType::ExtProtoInfo fpi;
    fpi.Variadic = false;
    llvm::ArrayRef<QualType> Args(ParamTypes);
    QualType newFT = ci.getASTContext().getFunctionType(
        ci.getASTContext().VoidTy, Args, fpi);

    // Then use FunctionDecl::Create() to create a new functiondecl
    auto fdecl = FunctionDecl::Create(ci.getASTContext(),
                                      R.getSema().getFunctionLevelDeclContext(),
                                      SourceLocation(), SourceLocation(),
                                      R.getLookupName(), newFT, 0, SC_None);
    std::vector<ParmVarDecl *> params{qb_copy, v_copy, h_copy};
    llvm::ArrayRef<ParmVarDecl *> parms(params);
    fdecl->setParams(parms);
    std::vector<Stmt *> svec;
    auto rtrn = ReturnStmt::CreateEmpty(ci.getASTContext(), false);
    svec.push_back(rtrn);
    llvm::ArrayRef<Stmt *> stmts(svec);
    auto cmp = CompoundStmt::Create(ci.getASTContext(), stmts, SourceLocation(),
                                    SourceLocation());
    fdecl->setBody(cmp);
    // fdecl->dump();

    R.addDecl(fdecl);
    return true;
  }
  return false;
}
} // namespace compiler
} // namespace qcor
