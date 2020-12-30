#pragma once
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wunused-function"

#include "ast/ast.hpp"
#include "ast/traversal.hpp"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "parser/parser.hpp"
#include "quantum_dialect.hpp"
#include "optimization/simplify.hpp"
#include "transformations/desugar.hpp"
#include "transformations/inline.hpp"

using namespace staq::ast;

namespace qasm_parser {

class StaqToMLIR : public staq::ast::Visitor {
 protected:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  std::map<std::string, mlir::quantum::QallocOp> qubit_allocations;

 public:
  StaqToMLIR(mlir::MLIRContext &context);
  mlir::ModuleOp module() {return theModule;}
  void visit(VarAccess &) override {}
  void visit(BExpr &) override {}
  void visit(UExpr &) override {}
  void visit(PiExpr &) override {}
  void visit(IntExpr &) override {}
  void visit(RealExpr &r) override {}
  void visit(VarExpr &v) override {}
  void visit(ResetStmt &) override {}
  void visit(IfStmt &) override {}
  void visit(BarrierGate &) override {}
  void visit(GateDecl &) override;
  void visit(OracleDecl &) override {}
  void visit(RegisterDecl &) override;
  void visit(AncillaDecl &) override {}
  void visit(Program &prog) override;
  void visit(MeasureStmt &m) override;
  void visit(UGate &u) override;
  void visit(CNOTGate &cx) override;
  void visit(DeclaredGate &g) override;
  void addReturn();
};
}  // namespace qasm_parser