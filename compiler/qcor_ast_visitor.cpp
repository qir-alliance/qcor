#include "qcor_ast_visitor.hpp"
#include "IRProvider.hpp"
#include "XACC.hpp"
#include "xacc_service.hpp"

// #include "qcor.hpp"
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

QCORASTVisitor::IsQuantumKernelVisitor::IsQuantumKernelVisitor(ASTContext &c)
    : context(c) {
  auto irProvider = xacc::getService<xacc::IRProvider>("quantum");
  validInstructions = irProvider->getInstructions();
  validInstructions.push_back("CX");

}

bool QCORASTVisitor::IsQuantumKernelVisitor::VisitDeclRefExpr(
    DeclRefExpr *expr) {
  if ( !foundSubLambda) {
    auto gateName = expr->getNameInfo().getAsString();
    if (std::find(validInstructions.begin(), validInstructions.end(),
                  gateName) != validInstructions.end()) {
      _isQuantumKernel = true;
    }
  }
  return true;
}

bool QCORASTVisitor::IsQuantumKernelVisitor::VisitLambdaExpr(LambdaExpr *expr) {
  foundSubLambda = true;
  return true;
}

QCORASTVisitor::QCORASTVisitor(CompilerInstance &c, Rewriter &rw)
    : ci(c), rewriter(rw) {}

bool QCORASTVisitor::VisitLambdaExpr(LambdaExpr *LE) {

  std::cout << "VISITING LAMBDA\n";
  // Get the Lambda Function Body
  Stmt *q_kernel_body = LE->getBody();

  // Double check... Is this a Quantum Kernel Lambda?
  IsQuantumKernelVisitor isqk(ci.getASTContext());
  isqk.TraverseStmt(LE->getBody());
    LE->dump();

  std::map<std::string, InstructionParameter> captures;
  // If it is, then map it to XACC IR
  if (isqk.isQuantumKernel()) {

    std::cout << "LAMBDA IS Quantum Kernel\n";
    // LE->dumpColor();
    // exit(0);
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
        // std::cout << varName << ", THIS DOUBLE VALUE IS KNOWN AT COMPILE
        // TIME: "
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

    SourceManager &SM = ci.getSourceManager();
    LangOptions &lo = ci.getLangOpts();
    lo.CPlusPlus11 = 1;
    auto xaccKernelLambdaStr =
        Lexer::getSourceText(CharSourceRange(LE->getSourceRange(), true), SM,
                             lo)
            .str();

    std::shared_ptr<Accelerator> targetAccelerator;
    if (xacc::optionExists("accelerator")) {
      targetAccelerator = xacc::getAccelerator();
    }

    std::cout << "LAMBDA STR:\n" << xaccKernelLambdaStr << "\n";
    auto compiler = xacc::getCompiler("xasm");
    auto ir = compiler->compile(xaccKernelLambdaStr, targetAccelerator);

    auto function = ir->getComposites()[0]; //.getFunction();
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

    auto sr = LE->getBody()->getSourceRange();
    if (!xacc::optionExists("accelerator")) {
      if (xacc::hasAccelerator("tnqvm")) {
        xacc::setOption("accelerator", "tnqvm");
      } else if (xacc::hasAccelerator("local-ibm")) {
        xacc::setOption("accelerator", "local-ibm");
      } else {
        xacc::error("No Accelerator specified for compilation. Compile with "
                    "--accelerator and ensure you have the desired Accelerator "
                    "installed.");
      }
    }

    auto acceleratorName = xacc::getAccelerator()->name();

    // Argument analysis
    // can be (qbit q, double t, double tt, double ttt, ...)
    // can be (qbit q, std::vector<double> t)

    auto F = LE->getCallOperator()->getAsFunction();
    auto bufferName = F->getParamDecl(0)->getNameAsString();
    int nArgs = F->getNumParams();

    // Rewrite the args list to have a boolean at the end
    // that is default to true and used to execute the accelerator
    // Also rewrite to (qbit q, std::vector<double> params)
    auto endOfParams = F->getParamDecl(nArgs-1)->getEndLoc().getLocWithOffset(1);
    rewriter.InsertText(endOfParams, ", bool execute = true");

    // Always write the Function to json string first
    std::stringstream ss;
    function->persist(ss);
    std::string replacement =
        "{\nauto irstr = R\"irstr(" + ss.str() + ")irstr\";\n";

    replacement += "if (execute) {\n";
    replacement += "auto function = xacc::getIRProvider(\"quantum\")->createComposite(\"f\");\n";
    replacement += "std::istringstream iss(ir);\n";
    replacement += "function->load(iss);\n";
    replacement +=
        "auto acc = xacc::getAccelerator(\"" + acceleratorName + "\");\n";
    if (F->getNumParams() > 1) {
      replacement +=
          "std::vector<double> params{" + F->getParamDecl(1)->getNameAsString();
      for (int i = 2; i < F->getNumParams(); i++) {
        replacement += "," + F->getParamDecl(i)->getNameAsString();
      }
      replacement += "};\n";
      // }
      // if (F->getNumParams() > 1) {
      replacement += "function = function->operator()(params);\n";
    }
    replacement += "acc->execute(" + bufferName + ",function);\n";
    replacement += "}\n";
    // }
    replacement += "return irstr;\n";
    replacement += "}\n";
    rewriter.ReplaceText(sr, replacement);

    // // Kick off quantum compilation
    // auto qcor = xacc::getCompiler("qcor");

    // std::shared_ptr<Accelerator> targetAccelerator;
    // if (xacc::optionExists("accelerator")) {
    //   targetAccelerator = xacc::getAccelerator();
    // }

    // function = qcor->compile(function, targetAccelerator);

    // auto fileName = qcor::persistCompiledCircuit(function,
    // targetAccelerator);

    // if (function->hasIRGenerators()) {
    //   auto begin = q_kernel_body->getBeginLoc();
    //   std::string replacement = "[&]() {\n";
    //   std::shared_ptr<Instruction> irg;
    //   for (auto i : function->getInstructions()) {
    //     if (std::dynamic_pointer_cast<IRGenerator>(i)) {
    //       irg = i;
    //       break;
    //     }
    //   }
    //   for (auto &kv : captures) {
    //     std::string key = "";
    //     auto opts = irg->getOptions();
    //     for (auto &kv2 : opts) {
    //       if (kv2.second.isVariable() && kv2.second.toString() == kv.first) {
    //         key = kv2.first;
    //       }
    //     }
    //     replacement +=
    //         "qcor::storeRuntimeVariable(\"" + key + "\", " + kv.first +
    //         ");\n";
    //   }
    //   replacement += "return \"" + fileName + "\";\n}";
    //   rewriter.ReplaceText(sr, replacement);

    // } else {
    //   rewriter.ReplaceText(sr, "[&](){return \"" + fileName + "\";}");
    // }
  }

  return true;
}

bool QCORASTVisitor::VisitFunctionDecl(FunctionDecl* decl) {
return true;
}
} // namespace compiler
} // namespace qcor
