#include "LambdaVisitor.hpp"
#include "/home/project/qcor/runtime/qcor.hpp"
#include "IRProvider.hpp"
#include "XACC.hpp"

#include "qcor.hpp"

using namespace clang;
using namespace xacc;

namespace qcor {
namespace compiler {

LambdaVisitor::IsQuantumKernelVisitor::IsQuantumKernelVisitor(ASTContext &c)
    : context(c) {
  auto irProvider = xacc::getService<xacc::IRProvider>("gate");
  validInstructions = irProvider->getInstructions();
  validInstructions.push_back("CX");
}

bool LambdaVisitor::IsQuantumKernelVisitor::VisitDeclRefExpr(
    DeclRefExpr *expr) {
  if (expr->getType() == context.DependentTy) {
    auto gateName = expr->getNameInfo().getAsString();
    if (std::find(validInstructions.begin(), validInstructions.end(),
                  gateName) != validInstructions.end()) {
      _isQuantumKernel = true;
    }
  }
}

LambdaVisitor::CppToXACCIRVisitor::CppToXACCIRVisitor(ASTContext &c)
    : context(c) {
  provider = xacc::getService<IRProvider>("gate");
  function = provider->createFunction("tmp", {});
}

bool LambdaVisitor::CppToXACCIRVisitor::VisitCallExpr(CallExpr *expr) {
  if (gateName != "") {
    // create the gate instruction
    if (gateName == "CX") {
      gateName = "CNOT";
    }

    auto inst = provider->createInstruction(gateName, bits, parameters);
    function->addInstruction(inst);
  }
  gateName = "";
  bits.clear();
  parameters.clear();
  return true;
}

bool LambdaVisitor::CppToXACCIRVisitor::VisitDeclRefExpr(DeclRefExpr *expr) {

  if (expr->getType() == context.DependentTy) {
    gateName = expr->getNameInfo().getAsString();

    //  std::cout << "VISITING " << gateName << ", " <<
    //  expr->getType().getAsString() << "\n";
  } else if (expr->getType() == context.DoubleTy) {
    InstructionParameter p(expr->getNameInfo().getAsString());
    parameters.push_back(p);
  }

  return true;
}

bool LambdaVisitor::CppToXACCIRVisitor::VisitIntegerLiteral(
    IntegerLiteral *literal) {
  bits.push_back(literal->getValue().getLimitedValue());
  return true;
}

bool LambdaVisitor::CppToXACCIRVisitor::VisitFloatingLiteral(
    FloatingLiteral *literal) {
  InstructionParameter p(literal->getValue().convertToDouble());
  parameters.push_back(p);
  return true;
}

std::shared_ptr<Function> LambdaVisitor::CppToXACCIRVisitor::getFunction() {
  // add the last one
  if (gateName != "") {
    // create the gate instruction
    if (gateName == "CX") {
      gateName = "CNOT";
    }

    auto inst = provider->createInstruction(gateName, bits, parameters);
    function->addInstruction(inst);
  }
  return function;
}
LambdaVisitor::LambdaVisitor(CompilerInstance &c) : ci(c) {}

bool LambdaVisitor::VisitLambdaExpr(LambdaExpr *LE) {

  // Get the Lambda Function Body
  Stmt *q_kernel_body = LE->getBody();

  // Double check... Is this a Quantum Kernel Lambda?
  IsQuantumKernelVisitor isqk(ci.getASTContext());
  isqk.TraverseStmt(LE->getBody());

  // If it is, then map it to XACC IR
  if (isqk.isQuantumKernel()) {

    // For debugging for now
    std::cout << "\n\n";
    q_kernel_body->dump();

    CppToXACCIRVisitor visitor(ci.getASTContext());
    visitor.TraverseStmt(LE->getBody());

    auto function = visitor.getFunction();
    std::cout << "\n\nXACC IR:\n" << function->toString() << "\n";

    // Kick off quantum compilation
    auto qcor = xacc::getCompiler("qcor");

    std::shared_ptr<Accelerator> targetAccelerator;
    if (xacc::optionExists("accelerator")) {
      targetAccelerator = xacc::getAccelerator();
    }

    function = qcor->compile(function, targetAccelerator);

    auto fileName = qcor::persistCompiledCircuit(function);

    // write function ir to file
    // update LE body to
    //   [](...) {
    //       return qcor::loadCompiledCircuit(filename);
    //   }
  }
  return true;
}

} // namespace compiler
} // namespace qcor
  //  LE->getType().dump();

// create empty statement, does nothing
// Stmt *tmp = (Stmt *)new (ci.getASTContext()) NullStmt(nopos);
// std::vector<Stmt *> stmts;
// stmts.push_back(tmp);
// replace LE's compound statement with that statement
// LE->getBody()->setLastStmt(ReturnStmt::CreateEmpty(
//     ci.getASTContext(),
//     false));

//   SourceManager &SM = ci.getSourceManager();
//   LangOptions &lo = ci.getLangOpts();
//   lo.CPlusPlus11 = 1;
//   auto xaccKernelLambdaStr =
//       Lexer::getSourceText(CharSourceRange(LE->getSourceRange(), true), SM,
//       lo)
//           .str();

//   std::cout << "Check it out, I got the Lambda as a source string :)\n";
//   xacc::info(xaccKernelLambdaStr);
