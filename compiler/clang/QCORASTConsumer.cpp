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

  using namespace std::chrono;
  auto start = std::chrono::high_resolution_clock::now();
  LambdaVisitor visitor(ci, rewriter);
  for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
    // if (std::string((*b)->getDeclKindName()) == "Function") {
    //   std::cout << (*b)->getDeclKindName() << "\n";
    //   (*b)->dumpColor();
      visitor.TraverseDecl(*b);
    // }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<microseconds>(stop - start);

//   std::cout << "Visitor time: " << duration.count() << ", " << std::endl;
  return true;
}
} // namespace compiler
} // namespace qcor