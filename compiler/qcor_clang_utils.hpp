#ifndef QCOR_COMPILER_CLANG_UTILS_HPP_
#define QCOR_COMPILER_CLANG_UTILS_HPP_

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTStructuralEquivalence.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "clang/AST/ASTImporter.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace qcor {
namespace compiler {
enum class DeclMatcherKind { First, Last };
enum class ExprMatcherKind { First, Last };

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

template <typename NodeType, ExprMatcherKind MatcherKind>
class ExprMatcher : public ast_matchers::MatchFinder::MatchCallback {
  NodeType *Node = nullptr;
  void run(const MatchFinder::MatchResult &Result) override {
    if ((MatcherKind == ExprMatcherKind::First && Node == nullptr) ||
        MatcherKind == ExprMatcherKind::Last) {
      Node = const_cast<NodeType *>(Result.Nodes.getNodeAs<NodeType>(""));
    }
  }
  ASTContext& _ctx;
public:
  ExprMatcher(ASTContext& ctx) : _ctx(ctx) {}
  // Returns the first/last matched node under the tree rooted in `D`.
  template <typename MatcherType>
  NodeType *match(const Expr *D, const MatcherType &AMatcher) {
    MatchFinder Finder;
    Finder.addMatcher(AMatcher.bind(""), this);
    Finder.matchAST(_ctx);
    assert(Node);
    return Node;
  }
};

template <typename NodeType>
using FirstDeclMatcher = DeclMatcher<NodeType, DeclMatcherKind::First>;
template <typename NodeType>
using FirstExprMatcher = ExprMatcher<NodeType, ExprMatcherKind::First>;
}
}
#endif