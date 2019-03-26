#include "LambdaVisitor.hpp"
#include "IRProvider.hpp"
#include "XACC.hpp"

#include "qcor.hpp"
#include "clang/Basic/SourceLocation.h"

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
  if (gateName == "Measure") {
    InstructionParameter p(bits[0]);
    parameters.push_back(p);
  }
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

    // Create the const char * QualType
    SourceLocation sl;
    QualType StrTy = ci.getASTContext().getConstantArrayType(
        ci.getASTContext().adjustStringLiteralBaseType(
            ci.getASTContext().CharTy.withConst()),
        llvm::APInt(32, fileName.length() + 1), ArrayType::Normal, 0);
    auto fnameSL = StringLiteral::Create(ci.getASTContext(), StringRef(fileName.c_str()),
                                      StringLiteral::Ascii,
                                      /*Pascal*/ false, StrTy, &sl, 1);

    // Create New Return type for CallOperator
    auto D = LE->getCallOperator()->getAsFunction();
    FunctionProtoType::ExtProtoInfo fpi;
    fpi.Variadic = D->isVariadic();
    std::vector<QualType> ParamTypes;
    llvm::ArrayRef<QualType> Args(ParamTypes);
    QualType newFT = D->getASTContext().getFunctionType(StrTy, Args, fpi);
    D->setType(newFT);

    // Create the return statement that will return
    // the string literal
    auto rtrn =
        ReturnStmt::Create(ci.getASTContext(), SourceLocation(), fnameSL, nullptr);
    std::vector<Stmt *> svec;
    svec.push_back(rtrn);
    llvm::ArrayRef<Stmt *> stmts(svec);
    auto cmp = CompoundStmt::Create(ci.getASTContext(), stmts, SourceLocation(),
                                    SourceLocation());
    LE->getCallOperator()->setBody(cmp);


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
