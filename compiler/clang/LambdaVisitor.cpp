#include "LambdaVisitor.hpp"
#include "IRGenerator.hpp"
#include "IRProvider.hpp"
#include "XACC.hpp"
#include "xacc_service.hpp"

#include "qcor.hpp"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Sema.h"
#include "clang/Tooling/Tooling.h"
#include <memory>

using namespace clang;
using namespace xacc;

namespace qcor {
namespace compiler {

LambdaVisitor::IsQuantumKernelVisitor::IsQuantumKernelVisitor(ASTContext &c)
    : context(c) {
  auto irProvider = xacc::getService<xacc::IRProvider>("quantum");
  validInstructions = irProvider->getInstructions();
  validInstructions.push_back("CX");
  auto irgens = xacc::getRegisteredIds<xacc::IRGenerator>();
  for (auto &irg : irgens) {
    validInstructions.push_back(irg);
  }
}

bool LambdaVisitor::IsQuantumKernelVisitor::VisitDeclRefExpr(
    DeclRefExpr *expr) {
  if (expr->getType() == context.DependentTy && !foundSubLambda) {
    auto gateName = expr->getNameInfo().getAsString();
    if (std::find(validInstructions.begin(), validInstructions.end(),
                  gateName) != validInstructions.end()) {
      _isQuantumKernel = true;
      if (irType != "anneal" && (gateName == "qmi" || gateName == "anneal")) {
          irType = "anneal";
      }
    }
  }
  return true;
}

bool LambdaVisitor::IsQuantumKernelVisitor::VisitLambdaExpr(LambdaExpr *expr) {
  foundSubLambda = true;
  return true;
}

LambdaVisitor::CppToXACCIRVisitor::CppToXACCIRVisitor(IsQuantumKernelVisitor& v) {
  provider = xacc::getService<IRProvider>("quantum");
  if (v.irType == "gate") {
    function = provider->createFunction("tmp", {}, {InstructionParameter("gate")});
  } else {
    function = provider->createFunction("tmp", {}, {InstructionParameter("anneal")});
  }

  auto irgens = xacc::getRegisteredIds<xacc::IRGenerator>();
  for (auto &irg : irgens) {
    irGeneratorNames.push_back(irg);
  }
}

bool LambdaVisitor::CppToXACCIRVisitor::VisitCallExpr(CallExpr *expr) {
  auto gate_name = dyn_cast<DeclRefExpr>(*(expr->child_begin()))
                       ->getNameInfo()
                       .getAsString();

  if (std::find(irGeneratorNames.begin(), irGeneratorNames.end(), gate_name) !=
      irGeneratorNames.end()) {

    // This is an IRGenerator
    // Map this CallExpr to an IRGenerator
    CallExprToIRGenerator visitor(gate_name, provider);
    visitor.TraverseStmt(expr);
    auto irg = visitor.getIRGenerator();
    if (irg->validateOptions()) {
      auto generated =
          irg->generate(std::map<std::string, InstructionParameter>{});
      for (auto inst : generated->getInstructions()) {
        function->addInstruction(inst);
      }
    } else {
      function->addInstruction(irg);
    }
  } else {
    // This is a regular gate
    // Map this Call Expr to a Instruction
    if (gate_name == "CX") {
      gate_name = "CNOT";
    }
    CallExprToXACCInstructionVisitor visitor(gate_name, provider);
    visitor.TraverseStmt(expr);
    auto inst = visitor.getInstruction();
    function->addInstruction(inst);
  }

  return true;
}

bool LambdaVisitor::CallExprToXACCInstructionVisitor::VisitIntegerLiteral(
    IntegerLiteral *il) {
  if (name == "anneal") {
      int i = il->getValue().getLimitedValue();
    InstructionParameter p(i);
    parameters.push_back(p);
  } else {
  bits.push_back(il->getValue().getLimitedValue());
  if (name == "Measure") {
    InstructionParameter p(bits[0]);
    parameters.push_back(p);
  }
  }
  return true;
}

bool LambdaVisitor::CallExprToXACCInstructionVisitor::VisitUnaryOperator(
    UnaryOperator *op) {
  if (op->getOpcode() == UnaryOperator::Opcode::UO_Minus) {
    addMinus = true;
  }
  return true;
}

bool LambdaVisitor::CallExprToXACCInstructionVisitor::VisitFloatingLiteral(
    FloatingLiteral *literal) {
  double value = literal->getValue().convertToDouble();
  InstructionParameter p(addMinus ? -1.0 * value : value);
  addMinus = false;
  parameters.push_back(p);
  return true;
}

bool LambdaVisitor::CallExprToXACCInstructionVisitor::VisitDeclRefExpr(
    DeclRefExpr *decl) {
  auto declName = decl->getNameInfo().getAsString();
  if (addMinus) {
    declName = "-" + declName;
  }
  if (dyn_cast<ParmVarDecl>(decl->getDecl())) {
    parameters.push_back(InstructionParameter(declName));
  } else if (dyn_cast<VarDecl>(decl->getDecl())) {
    // std::cout << "THIS IS A VARDECL: " << declName << "\n";
    parameters.push_back(InstructionParameter(declName));
  }
  return true;
}

std::shared_ptr<Instruction>
LambdaVisitor::CallExprToXACCInstructionVisitor::getInstruction() {
  return provider->createInstruction(name, bits, parameters);
}

bool LambdaVisitor::CallExprToIRGenerator::VisitInitListExpr(
    InitListExpr *expr) {
  if (haveSeenFirstInit && keepSearching) {

    for (auto child : immediate_children) {
      ScanInitListExpr visitor;
      visitor.TraverseStmt(child);
      options.insert({visitor.key, visitor.value});
      std::cout << "Inserting " << visitor.key << ", "
                << visitor.value.toString() << "\n";
    }

    keepSearching = false;

  } else {
    haveSeenFirstInit = true;
    auto children = expr->children();
    for (auto it = children.begin(); it != children.end(); ++it) {
      immediate_children.push_back(*it);
    }
  }
  return true;
}

bool LambdaVisitor::CallExprToIRGenerator::VisitDeclRefExpr(DeclRefExpr *decl) {

  if (dyn_cast<ParmVarDecl>(decl->getDecl())) {
    auto declName = decl->getNameInfo().getAsString();
    // std::cout << "IRGENERATOR FOUND PARAM: " << declName << "\n";
    options.insert({"param-id", declName});
  }
  return true;
}
std::shared_ptr<IRGenerator>
LambdaVisitor::CallExprToIRGenerator::getIRGenerator() {
  auto irg = xacc::getService<IRGenerator>(name);
  for (auto &kv : options) {
    irg->setOption(kv.first, kv.second);
  }
  return irg;
}

bool LambdaVisitor::ScanInitListExpr::VisitDeclRefExpr(DeclRefExpr *expr) {
  value = InstructionParameter(expr->getNameInfo().getAsString());
  return true;
}
bool LambdaVisitor::ScanInitListExpr::VisitInitListExpr(InitListExpr *expr) {

  if (skipSubInits) {
    return true;
  }

  if (hasSeenFirstIL) {
    HasSubInitListExpr visitor;
    visitor.TraverseStmt(*expr->children().begin());
    if (visitor.hasSubInitLists) {
      isVectorValue = true;
      // this is a vector of pairs or doubles.

      GetPairVisitor visitor;
      visitor.TraverseStmt(expr);
      if (!visitor.intsFound.empty()) {
        std::vector<std::pair<int, int>> tmp;
        for (int i = 0; i < visitor.intsFound.size(); i += 2) {
          tmp.push_back({visitor.intsFound[i], visitor.intsFound[i + 1]});
        }
        value = InstructionParameter(tmp);
      } else if (!visitor.realsFound.empty()) {
        std::vector<std::pair<double, double>> tmp;
        for (int i = 0; i < visitor.realsFound.size(); i += 2) {
          tmp.push_back({visitor.realsFound[i], visitor.realsFound[i + 1]});
        }
        value = InstructionParameter(tmp);
      } else {
        xacc::error("invalid vector<pair> type for IRGenerator options.");
      }

      skipSubInits = true;
    } else {

      // this is a vector...
      ScanInitListExpr visitor(true);
      visitor.TraverseStmt(expr);
      if (!visitor.intsFound.empty()) {
        value = InstructionParameter(visitor.intsFound);
      } else if (!visitor.realsFound.empty()) {
        value = InstructionParameter(visitor.realsFound);
      } else if (!visitor.stringsFound.empty()) {
        value = InstructionParameter(visitor.stringsFound);
      } else {
        xacc::error("invalid vector type for IRGenerator options.");
      }
    }
  } else {
    hasSeenFirstIL = true;
  }
  return true;
}

bool LambdaVisitor::ScanInitListExpr::VisitStringLiteral(
    StringLiteral *literal) {

  if (isVectorValue) {
    stringsFound.push_back(literal->getString().str());
  } else {
    if (isFirstStringLiteral) {
      isFirstStringLiteral = false;
      key = literal->getString().str();
    } else {
      value = InstructionParameter(literal->getString().str());
    }
  }
  return true;
}
bool LambdaVisitor::ScanInitListExpr::VisitFloatingLiteral(
    FloatingLiteral *literal) {

  if (isVectorValue) {

    realsFound.push_back(literal->getValue().convertToDouble());

  } else {
    value = InstructionParameter(literal->getValue().convertToDouble());
  }
  return true;
}
bool LambdaVisitor::ScanInitListExpr::VisitIntegerLiteral(
    IntegerLiteral *literal) {

  if (isVectorValue) {

    intsFound.push_back((int)literal->getValue().getLimitedValue());

  } else {
    value = InstructionParameter((int)literal->getValue().getLimitedValue());
  }
  return true;
}

bool LambdaVisitor::GetPairVisitor::VisitFloatingLiteral(
    FloatingLiteral *literal) {
  realsFound.push_back(literal->getValue().convertToDouble());
  return true;
}
bool LambdaVisitor::GetPairVisitor::VisitIntegerLiteral(
    IntegerLiteral *literal) {
  intsFound.push_back((int)literal->getValue().getLimitedValue());
  return true;
}

std::shared_ptr<Function> LambdaVisitor::CppToXACCIRVisitor::getFunction() {
  return function;
}

LambdaVisitor::LambdaVisitor(CompilerInstance &c, Rewriter &rw)
    : ci(c), rewriter(rw) {}

bool LambdaVisitor::VisitLambdaExpr(LambdaExpr *LE) {

  // Get the Lambda Function Body
  Stmt *q_kernel_body = LE->getBody();

  // Double check... Is this a Quantum Kernel Lambda?
  IsQuantumKernelVisitor isqk(ci.getASTContext());
  isqk.TraverseStmt(LE->getBody());
  //   LE->dump();

  std::map<std::string, InstructionParameter> captures;
  // If it is, then map it to XACC IR
  if (isqk.isQuantumKernel()) {

    LE->dumpColor();

    auto cb = LE->capture_begin(); // implicit_capture_begin();
    auto ce = LE->capture_end();
    VarDecl *v;
    for (auto it = cb; it != ce; ++it) {
      auto varName = it->getCapturedVar()->getNameAsString();

      //   it->getCapturedVar()->dumpColor();
      auto e = it->getCapturedVar()->getInit();
      auto int_value = dyn_cast<IntegerLiteral>(e);
      auto float_value = dyn_cast<FloatingLiteral>(e);
      if (int_value) {
        // std::cout << "THIS VALUE IS KNOWN AT COMPILE TIME: "
        //           << (int)int_value->getValue().signedRoundToDouble()
        //           << "\n"; // getAsString(ci.getASTContext(),
        //                    // it->getCapturedVar()->getType()) << "\n";
        captures.insert(
            {varName, (int)int_value->getValue().signedRoundToDouble()});
        continue;
      } else if (float_value) {
        // std::cout << varName << ", THIS DOUBLE VALUE IS KNOWN AT COMPILE TIME: "
        //           << float_value->getValue().convertToDouble() << "\n";
        captures.insert({varName, float_value->getValue().convertToDouble()});
        continue;
      }

      auto varType =
          it->getCapturedVar()->getType().getCanonicalType().getAsString();
      //   std::cout << "TYPE: " << varType << "\n";
      //   it->getCapturedVar()->dumpColor();
      captures.insert({varName, varName});
      //   v = it->getCapturedVar();
    }

    // q_kernel_body->dumpColor();
    CppToXACCIRVisitor visitor(isqk);
    visitor.TraverseStmt(LE);

    auto function = visitor.getFunction();
    for (auto &inst : function->getInstructions()) {
      if (!inst->isComposite() && inst->nParameters() > 0) {
        int counter = 0;
        for (auto &p : inst->getParameters()) {
          if (p.isVariable()) {
            // see if we have a runtime value in the captures map
            for (auto &kv : captures) {
              if (p.toString() == kv.first && kv.second.isNumeric()) {
                inst->setParameter(counter, kv.second);
              } else if (p.toString() == "-" + kv.first &&
                         kv.second.which() == 1) {
                InstructionParameter pp(-1.0 * mpark::get<double>(kv.second));
                inst->setParameter(counter, pp);
              }
            }
          }
          counter++;
        }
      }
    }

    std::cout << "\n\nXACC IR:\n" << function->toString() << "\n";

    // Check if we have IRGenerators in the tree
    if (function->hasIRGenerators()) {
      //   std::cout << "We have IRGenerators, checking to see if we know enough
      //   to "
      //    "generate it\n";
      int idx = 0;
      std::shared_ptr<IRGenerator> irg;
      for (auto &inst : function->getInstructions()) {
        irg = std::dynamic_pointer_cast<IRGenerator>(inst);
        if (irg) {
          for (auto &kv : captures) {
            if (kv.second.isNumeric()) {
              std::string key = "";
              auto opts = irg->getOptions();
              for (auto &kv2 : opts) {
                if (kv2.second.isVariable() &&
                    kv2.second.toString() == kv.first) {
                  key = kv2.first;
                }
              }
              if (!key.empty()) {
                irg->setOption(key, kv.second);
              }
            }
          }

          if (irg->validateOptions()) {
            function->expandIRGenerators(
                std::map<std::string, InstructionParameter>{});
          }
        }
        idx++;
      }
    }

    // std::cout << "\n\nAgain XACC IR:\n" << function->toString() << "\n";

    // Kick off quantum compilation
    auto qcor = xacc::getCompiler("qcor");

    std::shared_ptr<Accelerator> targetAccelerator;
    if (xacc::optionExists("accelerator")) {
      targetAccelerator = xacc::getAccelerator();
    }

    function = qcor->compile(function, targetAccelerator);

    auto fileName = qcor::persistCompiledCircuit(function, targetAccelerator);

    auto sr = LE->getSourceRange();

    if (function->hasIRGenerators()) {
      auto begin = q_kernel_body->getBeginLoc();
      std::string replacement = "[&]() {\n";
      std::shared_ptr<Instruction> irg;
      for (auto i : function->getInstructions()) {
        if (std::dynamic_pointer_cast<IRGenerator>(i)) {
          irg = i;
          break;
        }
      }
      for (auto &kv : captures) {
        std::string key = "";
        auto opts = irg->getOptions();
        for (auto &kv2 : opts) {
          if (kv2.second.isVariable() && kv2.second.toString() == kv.first) {
            key = kv2.first;
          }
        }
        replacement +=
            "qcor::storeRuntimeVariable(\"" + key + "\", " + kv.first + ");\n";
      }
      replacement += "return \"" + fileName + "\";\n}";
      rewriter.ReplaceText(sr, replacement);

    } else {
      rewriter.ReplaceText(sr, "[&](){return \"" + fileName + "\";}");
    }

    // Here we update the AST Node to change the
    // function prototype to string ()
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

    std::vector<Stmt *> svec;
    // Create the return statement that will return
    // the string literal file name
    auto rtrn = ReturnStmt::Create(ci.getASTContext(), SourceLocation(),
                                   fnameSL, nullptr);

    auto cs = LE->getCallOperator()->getBody();
    for (auto it = cs->child_begin(); it != cs->child_end(); ++it) {
      svec.push_back(*it);
    }

    // svec.push_back(LE->getCallOperator()->getBody());
    svec.push_back(rtrn);

    llvm::ArrayRef<Stmt *> stmts(svec);
    auto cmp = CompoundStmt::Create(ci.getASTContext(), stmts, SourceLocation(),
                                    SourceLocation());
    LE->getCallOperator()->setBody(cmp);
    // LE->getCallOperator()->dumpColor();

    // std::cout << "LE DUMP\n";
    // LE->dumpColor();
  }

  return true;
}

} // namespace compiler
} // namespace qcor
  //  LE->getType().dump();

// Create the const char * QualType
// SourceLocation sl;
// QualType StrTy = ci.getASTContext().getConstantArrayType(
//     ci.getASTContext().adjustStringLiteralBaseType(
//         ci.getASTContext().CharTy.withConst()),
//     llvm::APInt(32, fileName.length() + 1), ArrayType::Normal, 0);
// auto fnameSL =
//     StringLiteral::Create(ci.getASTContext(), StringRef(fileName.c_str()),
//                           StringLiteral::Ascii, false, StrTy, &sl, 1);

// // // Create New Return type for CallOperator
// std::vector<QualType> ParamTypes;
// auto D = LE->getCallOperator()->getAsFunction();
// FunctionProtoType::ExtProtoInfo fpi;
// fpi.Variadic = D->isVariadic();
// llvm::ArrayRef<QualType> Args(ParamTypes);
// QualType newFT = D->getASTContext().getFunctionType(StrTy, Args, fpi);
// D->setType(newFT);

// /*
//    Here we need to create instructions that add capture variables
//    to a runtime parameter map, so that when this lambda is called,
//    any runtime valued variables are added to the map and available
//    on the qcor runtime side.
// */

// std::vector<Stmt *> svec;

// auto cb = LE->implicit_capture_begin();
// auto ce = LE->implicit_capture_end();
// for (auto it = cb; it != ce; ++it) {

//   it->getCapturedVar()->dump();
//   auto e = it->getCapturedVar()->getInit();
//   auto value = dyn_cast<IntegerLiteral>(e);
//   if (value) {
//     std::cout << "THIS VALUE IS KNOWN AT COMPILE TIME: "
//               << (int)value->getValue().signedRoundToDouble()
//               << "\n"; // getAsString(ci.getASTContext(),
//                        // it->getCapturedVar()->getType()) << "\n";
//   }
//   auto varName = it->getCapturedVar()->getNameAsString();
//   auto varType =
//       it->getCapturedVar()->getType().getCanonicalType().getAsString();
//   std::cout << "TYPE: " << varType << "\n";
// }

// Create the return statement that will return
// the string literal file name
// auto rtrn = ReturnStmt::Create(ci.getASTContext(), SourceLocation(),
//                                fnameSL, nullptr);

// svec.push_back(rtrn);

// llvm::ArrayRef<Stmt *> stmts(svec);
// auto cmp = CompoundStmt::Create(ci.getASTContext(), stmts, SourceLocation(),
//                                 SourceLocation());
// LE->getCallOperator()->setBody(cmp);
// LE->getCallOperator()->dump();
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
