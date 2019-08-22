#include "fuzzy_parsing.hpp"

#include "IRProvider.hpp"
#include "XACC.hpp"
#include "xacc_service.hpp"

#include "Utils.hpp"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/AST/ASTStructuralEquivalence.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "clang/AST/ASTImporter.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace qcor {
namespace compiler {
enum class DeclMatcherKind { First, Last };

// Matcher class to retrieve the first/last matched node under a given AST.
template <typename NodeType, DeclMatcherKind MatcherKind>
class DeclMatcher : public ast_matchers::MatchFinder::MatchCallback {
  NodeType *Node = nullptr;
  void run(const MatchFinder::MatchResult &Result) override {
    if ((MatcherKind == DeclMatcherKind::First && Node == nullptr) ||
        MatcherKind == DeclMatcherKind::Last) {
      Node = const_cast<NodeType *>(Result.Nodes.getNodeAs<NodeType>(""));
    }
  }
public:
  // Returns the first/last matched node under the tree rooted in `D`.
  template <typename MatcherType>
  NodeType *match(const Decl *D, const MatcherType &AMatcher) {
    MatchFinder Finder;
    Finder.addMatcher(AMatcher.bind(""), this);
    Finder.matchAST(D->getASTContext());
    assert(Node);
    return Node;
  }
};

template <typename NodeType>
using FirstDeclMatcher = DeclMatcher<NodeType, DeclMatcherKind::First>;

void FuzzyParsingExternalSemaSource::initialize() {
  auto irProvider = xacc::getService<xacc::IRProvider>("quantum");
  validInstructions = irProvider->getInstructions();
  validInstructions.push_back("CX");
}

bool FuzzyParsingExternalSemaSource::LookupUnqualified(clang::LookupResult &R,
                                                       clang::Scope *S) {
  DeclarationName Name = R.getLookupName();
  std::string unknownName = Name.getAsString();

  // If this is a valid quantum instruction, tell Clang its
  // all gonna be ok, we got this...
  if (std::find(validInstructions.begin(),
                validInstructions.end(), // not template scope
                unknownName) != validInstructions.end() &&
      S->getFlags() != 128 && S->getBlockParent() != nullptr) {

    std::cout << "HELLO FP: " << unknownName << ", " << S->getFlags() << "\n";

    IdentifierInfo *II = Name.getAsIdentifierInfo();
    SourceLocation Loc = R.getNameLoc();


    std::string src = "";
   if (unknownName == "H" || unknownName == "Measure") {
      src ="void "+unknownName+"(int q) {return;}";
   } else if (unknownName == "CX") {
       src = "void "+unknownName+"(int q, int q2){return;}";
   }
    auto AST = tooling::buildASTFromCodeWithArgs( src, {"-std=c++11"});
    ASTs.push_back(std::move(AST));

    auto Matcher = namedDecl(hasName(unknownName));

FunctionDecl *D0 = FirstDeclMatcher<FunctionDecl>().match(
        ASTs[ASTs.size()-1]->getASTContext().getTranslationUnitDecl(), Matcher);

D0->setDeclContext(R.getSema().getFunctionLevelDeclContext());

//  FunctionDecl *copy =FunctionDecl::Create(
//         *m_Context, R.getSema().getFunctionLevelDeclContext(), Loc, Loc, Name,
//         m_Context->DependentTy, 0, SC_None);
// std::memcpy(copy, D0, sizeof(FunctionDecl));
    // auto fdecl = FunctionDecl::Create(
    //     *m_Context, R.getSema().getFunctionLevelDeclContext(), Loc, Loc, Name,
    //     m_Context->DependentTy, 0, SC_None);

    // // xacc::print_backtrace();

    // Stmt *S = new (*m_Context) NullStmt(Stmt::EmptyShell());
    // fdecl->setBody(S);

     D0->dump();
    R.addDecl(D0);


    // fdecl->getReturnType()->isUndeducedType()
    // std::cout <<  "ISDELETED: " << std::boolalpha << fdecl->isDeleted() << ", " << fdecl->getReturnType()->isUndeducedType() << "\n";
    return true;
  }
  return false;
}
} // namespace compiler
} // namespace qcor
    // VarDecl *Result =
    //     VarDecl:Create(*m_Context, R.getSema().getFunctionLevelDeclContext(),
    //                     Loc, Loc, II, m_Context->DependentTy,
    //                     /*TypeSourceInfo*/ 0, SC_None);
    // Create the const char * QualType
    // SourceLocation sl;
    // QualType StrTy = m_Context->getConstantArrayType(
    //     m_Context->adjustStringLiteralBaseType(m_Context->CharTy.withConst()),
    //     llvm::APInt(32, 1), ArrayType::Normal, 0);
    // auto fnameSL = StringLiteral::Create(
    //     *m_Context, StringRef(""), StringLiteral::Ascii, false, StrTy, &sl, 1);

    // // Create the return statement that will return
    // // the string literal file name
    // auto rtrn =
    //     ReturnStmt::Create(*m_Context, SourceLocation(), fnameSL, nullptr);
    // std::vector<Stmt *> svec;

    // svec.push_back(rtrn);

    // llvm::ArrayRef<Stmt *> stmts(svec);
    // auto cmp = CompoundStmt::Create(*m_Context, stmts, SourceLocation(),
    //                                 SourceLocation());
