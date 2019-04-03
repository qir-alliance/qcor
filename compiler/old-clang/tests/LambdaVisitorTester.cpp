#include <gtest/gtest.h>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/MemoryBuffer.h"

#include "FuzzyParsingExternalSemaSource.hpp"
#include "LambdaVisitor.hpp"
#include "QCORASTConsumer.hpp"
#include "QCORPluginAction.hpp"

#include "XACC.hpp"
#include "qcor.hpp"
#include "clang/Parse/ParseAST.h"

using namespace llvm;
using namespace clang;
using namespace qcor;

class TestQCORFrontendAction : public clang::ASTFrontendAction {

protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef /* dummy */) override {
    return llvm::make_unique<compiler::QCORASTConsumer>(Compiler);
  }

  void ExecuteAction() override {
    CompilerInstance &CI = getCompilerInstance();
    CI.createSema(getTranslationUnitKind(), nullptr);
    CI.getSema().addExternalSource(
        new compiler::FuzzyParsingExternalSemaSource(CI.getASTContext()));

    ParseAST(CI.getSema());
  }

};

const std::string bell = R"bell(
void foo(const char * s, int t) {return;}
int main() {
    foo("name", 3);
    int a = 3;
    auto l = [&a]() {
        H(0);
        CX(0,1);
        Measure(0);
    };
    return 0;
})bell";

const std::string param0 = R"param0(int main() {
    auto l = [&](double t) {
        X(0);
        Ry(t,1);
        CX(1,0);
    };
    return 0;
})param0";

const std::string hwe0 = R"hwe0(#include <vector>
int main(int argc, char** argv){
    int nq = 2;
    auto l = [&](std::vector<double> x) {
        hwe(x, nq, {{"layers",1}});
    };
    return 0;
})hwe0";

const std::string hwe1 = R"hwe1(#include <vector>
int main(int argc, char** argv){
    int nq = argc;
    auto l = [&](std::vector<double> x) {
        hwe(x, nq, {{"layers",1}});
        Measure(0);
    };
    return 0;
})hwe1";

TEST(LambdaVisitorTester, checkSimple) {
  auto actionBell = new TestQCORFrontendAction();
  auto actionParam0 = new TestQCORFrontendAction();
  std::vector<std::string> args{"-std=c++11"};

  EXPECT_TRUE(tooling::runToolOnCodeWithArgs(
      actionBell, bell,
      args));

  EXPECT_TRUE(tooling::runToolOnCodeWithArgs(
      actionParam0, param0,
      args));
}

TEST(LambdaVisitorTester, checkGenerator) {
  auto action = new TestQCORFrontendAction();
  std::vector<std::string> args{"-std=c++11"};
  EXPECT_TRUE(tooling::runToolOnCodeWithArgs(
      action, hwe0,
      args));

  auto action2 = new TestQCORFrontendAction();
  EXPECT_TRUE(tooling::runToolOnCodeWithArgs(
      action2, hwe1,
      args));
}

int main(int argc, char **argv) {
  qcor::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
