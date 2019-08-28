#include "fuzzy_parsing.hpp"

#include "IRProvider.hpp"
#include "XACC.hpp"
#include "xacc_service.hpp"

#include "Utils.hpp"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTImporter.h"

#include "qcor_clang_utils.hpp"

using namespace clang;
using namespace clang::ast_matchers;

namespace qcor {
namespace compiler {

void FuzzyParsingExternalSemaSource::initialize() {
  auto provider = xacc::getService<xacc::IRProvider>("quantum");
  validInstructions = provider->getInstructions();
  validInstructions.push_back("CX");

  for (auto &instructionName : validInstructions) {
    std::string tmpSource = "void " + instructionName + "(";
    auto tmpInst = provider->createInstruction(
        instructionName == "CX" ? "CNOT" : instructionName, {});
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
    quantumInstructionASTs.insert(
        {instructionName + "__qcor_instruction",
         tooling::buildASTFromCodeWithArgs(tmpSource, {"-std=c++11"})});
  }
}

bool FuzzyParsingExternalSemaSource::LookupUnqualified(clang::LookupResult &R,
                                                       clang::Scope *S) {
  std::string unknownName = R.getLookupName().getAsString();

  // If this is a valid quantum instruction, tell Clang its
  // all gonna be ok, we got this...
  if (quantumInstructionASTs.count(unknownName + "__qcor_instruction") &&
      S->getFlags() != 128 && S->getBlockParent() != nullptr) {

    auto Matcher = namedDecl(hasName(unknownName));

    FunctionDecl *D0 = FirstDeclMatcher<FunctionDecl>().match(
        quantumInstructionASTs[unknownName + "__qcor_instruction"]
            ->getASTContext()
            .getTranslationUnitDecl(),
        Matcher);

    R.addDecl(D0);

    return true;
  }
  return false;
}
} // namespace compiler
} // namespace qcor
