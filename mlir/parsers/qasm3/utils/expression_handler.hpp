#pragma once

#include "Quantum/QuantumOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "qasm3BaseVisitor.h"
#include "qasm3Parser.h"
#include "qasm3_utils.hpp"
#include "symbol_table.hpp"

static constexpr double pi = 3.141592653589793238;
using namespace qasm3;

namespace qcor {
class qasm3_expression_generator : public qasm3::qasm3BaseVisitor {
 protected:
  mlir::OpBuilder builder;
  mlir::ModuleOp m_module;
  std::string file_name = "";
  // std::map<std::string, mlir::Value>& global_symbol_table;
  bool next_value_is_idx = false;
  bool casting_indexed_integer_to_bool = false;
  
  mlir::Value indexed_variable_value;

  bool is_signed = true;
  std::size_t number_width;
  mlir::Type internal_float_type;
  mlir::Type internal_integer_type;

  std::size_t current_idx = -1;

  ScopedSymbolTable& symbol_table;
  mlir::Value create_constant_integer_value(const std::size_t idx,
                                            mlir::Location location);
  mlir::Value get_or_extract_qubit(const std::string& qreg_name,
                                   const std::size_t idx,
                                   mlir::Location location);

  mlir::Value get_or_create_constant_integer_value(const std::size_t idx,
                                                   mlir::Location location,
                                                   int width = 64);

  mlir::Value get_or_create_constant_index_value(const std::size_t idx,
                                                 mlir::Location location,
                                                 int width = 64);

  void update_current_value(mlir::Value v);

  template <typename OpTy, typename... Args>
  OpTy createOp(Args... args) {
    OpTy value = builder.create<OpTy>(args...);
    update_current_value(value);
    return value;
  }

 public:
  mlir::Value current_value;
  mlir::Value last_current_value;

  qasm3_expression_generator(mlir::OpBuilder b, ScopedSymbolTable& table,
                             std::string& fname, std::size_t nw = 64, bool _is_signed = true);

  antlrcpp::Any visitTerminal(antlr4::tree::TerminalNode* node) override;
  antlrcpp::Any visitExpression(qasm3Parser::ExpressionContext* ctx) override;

  antlrcpp::Any visitIncrementor(qasm3Parser::IncrementorContext* ctx) override;

  antlrcpp::Any visitExpressionTerminator(
      qasm3Parser::ExpressionTerminatorContext* ctx) override;
};
}  // namespace qcor