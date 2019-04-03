#include "LambdaVisitor.hpp"
#include "IRGenerator.hpp"
#include "IRProvider.hpp"
#include "XACC.hpp"

#include "qcor.hpp"
#include "clang/AST/Decl.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Sema.h"
#include "clang/Tooling/Tooling.h"

using namespace clang;
using namespace xacc;

namespace qcor {
namespace compiler {

LambdaVisitor::IsQuantumKernelVisitor::IsQuantumKernelVisitor(ASTContext &c)
    : context(c) {
  auto irProvider = xacc::getService<xacc::IRProvider>("gate");
  validInstructions = irProvider->getInstructions();
  validInstructions.push_back("CX");
  auto irgens = xacc::getRegisteredIds<xacc::IRGenerator>();
  for (auto &irg : irgens) {
    validInstructions.push_back(irg);
  }
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

  auto irgens = xacc::getRegisteredIds<xacc::IRGenerator>();
  for (auto &irg : irgens) {
    irGeneratorNames.push_back(irg);
  }
}

bool LambdaVisitor::CppToXACCIRVisitor::VisitCallExpr(CallExpr *expr) {
  if (gateName != "") {
    // create the gate instruction
    if (gateName == "CX") {
      gateName = "CNOT";
    }

    if (std::find(irGeneratorNames.begin(), irGeneratorNames.end(), gateName) !=
        irGeneratorNames.end()) {
      std::cout << "This is an IR Generator\n";
      auto irg = xacc::getService<IRGenerator>(gateName);
      function->addInstruction(irg);

    } else {
      auto inst = provider->createInstruction(gateName, bits, parameters);
      function->addInstruction(inst);
    }
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
bool LambdaVisitor::CppToXACCIRVisitor::VisitParmVarDecl(ParmVarDecl *decl) {
  //   std::cout << "FOUND PARM VAR DECL, " << decl->getNameAsString() << "\n";
  parameters.push_back(InstructionParameter(decl->getNameAsString()));
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

    if (std::find(irGeneratorNames.begin(), irGeneratorNames.end(), gateName) !=
        irGeneratorNames.end()) {
      std::cout << "This is an IR Generator " << gateName << "\n";
      auto irg = xacc::getService<IRGenerator>(gateName);
      function->addInstruction(irg);
    } else {
      auto inst = provider->createInstruction(gateName, bits, parameters);
      function->addInstruction(inst);
    }
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
  //   LE->dump();

  // If it is, then map it to XACC IR
  if (isqk.isQuantumKernel()) {

    // For debugging for now
    // std::cout << "\n\n";
    // q_kernel_body->dump();

    CppToXACCIRVisitor visitor(ci.getASTContext());
    visitor.TraverseStmt(LE);

    auto function = visitor.getFunction();
    std::cout << "\n\nXACC IR:\n" << function->toString() << "\n";

    // Kick off quantum compilation
    auto qcor = xacc::getCompiler("qcor");

    std::shared_ptr<Accelerator> targetAccelerator;
    if (xacc::optionExists("accelerator")) {
      targetAccelerator = xacc::getAccelerator();
    }

    function = qcor->compile(function, targetAccelerator);

    auto fileName = qcor::persistCompiledCircuit(function, targetAccelerator);

    // Create the const char * QualType
    SourceLocation sl;
    QualType StrTy = ci.getASTContext().getConstantArrayType(
        ci.getASTContext().adjustStringLiteralBaseType(
            ci.getASTContext().CharTy.withConst()),
        llvm::APInt(32, fileName.length() + 1), ArrayType::Normal, 0);
    auto fnameSL =
        StringLiteral::Create(ci.getASTContext(), StringRef(fileName.c_str()),
                              StringLiteral::Ascii, false, StrTy, &sl, 1);

    // Create New Return type for CallOperator
    std::vector<QualType> ParamTypes;
    auto D = LE->getCallOperator()->getAsFunction();
    FunctionProtoType::ExtProtoInfo fpi;
    fpi.Variadic = D->isVariadic();
    llvm::ArrayRef<QualType> Args(ParamTypes);
    QualType newFT = D->getASTContext().getFunctionType(StrTy, Args, fpi);
    D->setType(newFT);

    /*
       Here we need to create instructions that add capture variables
       to a runtime parameter map, so that when this lambda is called,
       any runtime valued variables are added to the map and available
       on the qcor runtime side.
    */

    std::vector<Stmt *> svec;

    auto cb = LE->implicit_capture_begin();
    auto ce = LE->implicit_capture_end();
    for (auto it = cb; it != ce; ++it) {

      it->getCapturedVar()->dump();
      auto e = it->getCapturedVar()->getInit();
      auto value = dyn_cast<IntegerLiteral>(e);
      if (value) {
        std::cout << "THIS VALUE IS KNOWN AT COMPILE TIME: "
                  << (int)value->getValue().signedRoundToDouble()
                  << "\n"; // getAsString(ci.getASTContext(),
                           // it->getCapturedVar()->getType()) << "\n";
      }
      auto varName = it->getCapturedVar()->getNameAsString();
      auto varType =
          it->getCapturedVar()->getType().getCanonicalType().getAsString();
      std::cout << "TYPE: " << varType << "\n";
      auto action = new ASTGeneratorAction();
      std::vector<std::string> args{"-std=c++11"};
      auto src = "#include <fstream>\nint main() {\n" + varType + " " +
                 varName +
                 ";\nauto l = [&]() {\n"
                 "std::ofstream os;\nos.open(\"out.txt\");\nos << \"" +
                 varName + ":\" << " + varName +
                 " << \"\\n\";\nos.close();\n};\nreturn 0;\n}";
      std::cout << "SRC:\n" << src << "\n";
      auto ast = tooling::buildASTFromCodeWithArgs(src, args);
      auto tl = ast->getASTContext().getTranslationUnitDecl();
      CallExprCloner cev;
      cev.TraverseDecl(tl);

      svec.push_back(cev.cloned->getCallOperator()->getAsFunction()->getBody());
    }

    // Create the return statement that will return
    // the string literal file name
    auto rtrn = ReturnStmt::Create(ci.getASTContext(), SourceLocation(),
                                   fnameSL, nullptr);

    svec.push_back(rtrn);

    llvm::ArrayRef<Stmt *> stmts(svec);
    auto cmp = CompoundStmt::Create(ci.getASTContext(), stmts, SourceLocation(),
                                    SourceLocation());
    LE->getCallOperator()->setBody(cmp);
    LE->getCallOperator()->dump();
    exit(0);
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
