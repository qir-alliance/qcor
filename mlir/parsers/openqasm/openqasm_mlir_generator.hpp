#pragma once

// Turn off Staq warnings
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wunused-function"

#include "ast/ast.hpp"
#include "ast/traversal.hpp"
#include "mlir/IR/Region.h"
#include "mlir_generator.hpp"
#include "optimization/simplify.hpp"
#include "parser/parser.hpp"
#include "quantum_dialect.hpp"
#include "transformations/desugar.hpp"
#include "transformations/inline.hpp"

using namespace staq::ast;

namespace qcor {

class OpenQasmMLIRGenerator : public qcor::QuantumMLIRGenerator,
                              public staq::ast::Visitor {
 protected:
  std::map<std::string, mlir::quantum::QallocOp> qubit_allocations;
  bool in_sub_kernel = false;
  std::map<std::string, mlir::Value> temporary_sub_kernel_args;
  bool is_first_inst = true;
  bool add_main = true;
  std::string file_name = "main";
  bool add_entry_point = true;
  mlir::Type qubit_type;
  mlir::Type array_type;
  mlir::Type result_type;

  std::map<std::pair<std::string, std::uint64_t>, mlir::Value> extracted_qubits;

 public:
  OpenQasmMLIRGenerator(mlir::MLIRContext &context)
      : QuantumMLIRGenerator(context) {}
  void initialize_mlirgen(bool add_entry_point = true,
                          const std::string file_name = "") override;
  void mlirgen(const std::string &src) override;
  void finalize_mlirgen() override;

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
class CountGateDecls : public staq::ast::Visitor {
 private:
  std::size_t &count;

 public:
  std::vector<std::string> gates_to_inline;

  CountGateDecls(std::size_t &c) : count(c) {}
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
  void visit(GateDecl &g) override;
  void visit(OracleDecl &) override {}
  void visit(RegisterDecl &) override {}
  void visit(AncillaDecl &) override {}
  void visit(Program &prog) override {}
  void visit(MeasureStmt &m) override {}
  void visit(UGate &u) override {}
  void visit(CNOTGate &cx) override {}
  void visit(DeclaredGate &g) override {}
};
}  // namespace qcor